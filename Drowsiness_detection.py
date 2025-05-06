import cv2
import numpy as np
from scipy.spatial import distance
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EAR_THRESHOLD_RATIO = 0.8
HEAD_POSE_THRESHOLD = 15
CALIBRATION_DURATION = 5
FRAME_HISTORY = 30
EMA_ALPHA = 0.05
MIN_FACE_CONFIDENCE = 0.5

# Landmark indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

class TemporalAnalyzer:
    def __init__(self, window_size=30):
        try:
            self.window_size = window_size
            self.ear_history = deque(maxlen=window_size)
            self.pose_history = deque(maxlen=window_size)
            self.blink_history = deque(maxlen=window_size)
            self.timestamps = deque(maxlen=window_size)
            logger.info("TemporalAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TemporalAnalyzer: {str(e)}")
            raise

    def update(self, ear, pose_angles):
        try:
            if not isinstance(ear, (int, float)) or not isinstance(pose_angles, (list, np.ndarray)):
                raise ValueError("Invalid input types for update")
            
            self.ear_history.append(ear)
            self.pose_history.append(pose_angles)
            self.timestamps.append(time.time())
            
            if len(self.ear_history) >= 2 and ear < 0.5 * self.ear_history[-2]:
                self.blink_history.append(1)
            else:
                self.blink_history.append(0)
        except Exception as e:
            logger.error(f"Error in TemporalAnalyzer.update: {str(e)}")

    def get_metrics(self):
        try:
            if len(self.ear_history) < 2:
                return {}
                
            ear_mean = np.mean(self.ear_history)
            ear_std = np.std(self.ear_history)
            blink_rate = (sum(self.blink_history) / len(self.blink_history)) if self.blink_history else 0
            
            pose_changes = []
            for i in range(1, len(self.pose_history)):
                if i < len(self.pose_history):
                    change = np.abs(np.array(self.pose_history[i]) - np.array(self.pose_history[i-1]))
                    pose_changes.append(np.mean(change))
            
            head_movement = np.mean(pose_changes) if pose_changes else 0
            
            return {
                'ear_mean': ear_mean,
                'ear_std': ear_std,
                'blink_rate': blink_rate,
                'head_movement': head_movement,
                'stability': 1.0 / (ear_std + 0.1) if ear_std != 0 else 1.0
            }
        except Exception as e:
            logger.error(f"Error in TemporalAnalyzer.get_metrics: {str(e)}")
            return {}

class AdaptiveThreshold:
    def __init__(self):
        try:
            self.ear_baseline = 0.3
            self.ear_threshold = 0.25
            self.pose_baseline = np.zeros(3)
            self.calibrated = False
            logger.info("AdaptiveThreshold initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AdaptiveThreshold: {str(e)}")
            raise

    def update_baselines(self, ear, pose_angles):
        try:
            if not isinstance(ear, (int, float)) or not isinstance(pose_angles, (list, np.ndarray)):
                raise ValueError("Invalid input types for update_baselines")
            
            if not self.calibrated:
                self.ear_baseline = ear
                self.pose_baseline = pose_angles
                self.calibrated = True
            else:
                self.ear_baseline = (1-EMA_ALPHA) * self.ear_baseline + EMA_ALPHA * ear
                self.ear_threshold = EAR_THRESHOLD_RATIO * self.ear_baseline
                self.pose_baseline = (1-EMA_ALPHA) * np.array(self.pose_baseline) + EMA_ALPHA * np.array(pose_angles)
        except Exception as e:
            logger.error(f"Error in AdaptiveThreshold.update_baselines: {str(e)}")

class DrowsinessDetector:
    def __init__(self):
        try:
            self.temporal_analyzer = TemporalAnalyzer(FRAME_HISTORY)
            self.adaptive_threshold = AdaptiveThreshold()
            self.state = "CALIBRATING"
            self.calibration_start = None
            self.last_alert_time = 0
            self.alert_cooldown = 5
            logger.info("DrowsinessDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DrowsinessDetector: {str(e)}")
            raise

    def calibrate(self, frame_count, duration):
        try:
            if self.calibration_start is None:
                self.calibration_start = time.time()
                
            remaining = duration - (time.time() - self.calibration_start)
            if remaining <= 0:
                self.state = "MONITORING"
                return True
            return False
        except Exception as e:
            logger.error(f"Error in DrowsinessDetector.calibrate: {str(e)}")
            return False

    def detect(self, eye_metrics, pose_angles):
        try:
            if not isinstance(eye_metrics, dict) or not isinstance(pose_angles, (list, np.ndarray)):
                raise ValueError("Invalid input types for detect")
            
            self.temporal_analyzer.update(eye_metrics.get('ear', 0), pose_angles)
            
            if self.state == "MONITORING":
                self.adaptive_threshold.update_baselines(eye_metrics.get('ear', 0), pose_angles)
            
            temp_metrics = self.temporal_analyzer.get_metrics()
            
            eye_score = 0
            if 'ear' in eye_metrics and eye_metrics['ear'] < self.adaptive_threshold.ear_threshold:
                eye_score = 1 - (eye_metrics['ear'] / self.adaptive_threshold.ear_threshold)
            elif eye_metrics.get('closure_ratio', 0) > 0.7:
                eye_score = eye_metrics['closure_ratio']
                
            pose_score = 0
            pitch_diff = abs(pose_angles[0] - self.adaptive_threshold.pose_baseline[0])
            if pitch_diff > HEAD_POSE_THRESHOLD:
                pose_score = min(1.0, pitch_diff / (2 * HEAD_POSE_THRESHOLD))
                
            temp_score = 0
            if temp_metrics.get('ear_mean', 0) < self.adaptive_threshold.ear_threshold:
                temp_score += 0.5 * (1 - temp_metrics['ear_mean'] / self.adaptive_threshold.ear_threshold)
            if temp_metrics.get('blink_rate', 0) < 0.1:
                temp_score += 0.2
            elif temp_metrics.get('blink_rate', 0) > 0.4:
                temp_score += 0.3
            if temp_metrics.get('head_movement', 1) < 1.0:
                temp_score += 0.2
                
            combined_score = (0.5 * eye_score + 0.3 * pose_score + 0.2 * temp_score)
            
            current_time = time.time()
            if self.state == "MONITORING":
                if combined_score > 0.7:
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        self.state = "ALERT"
                        self.last_alert_time = current_time
                elif combined_score > 0.5:
                    self.state = "WARNING"
                else:
                    self.state = "NORMAL"
            elif self.state == "ALERT" and combined_score < 0.4:
                self.state = "NORMAL"
                
            return {
                'state': self.state,
                'score': combined_score,
                'components': {
                    'eyes': eye_score,
                    'pose': pose_score,
                    'temporal': temp_score
                },
                'metrics': {
                    'ear': eye_metrics.get('ear', 0),
                    'ear_threshold': self.adaptive_threshold.ear_threshold,
                    'pitch': pose_angles[0],
                    'blink_rate': temp_metrics.get('blink_rate', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error in DrowsinessDetector.detect: {str(e)}")
            return {
                'state': 'ERROR',
                'score': 0,
                'components': {'eyes': 0, 'pose': 0, 'temporal': 0},
                'metrics': {'ear': 0, 'ear_threshold': 0.25, 'pitch': 0, 'blink_rate': 0}
            }

def eye_aspect_ratio(eye_landmarks):
    try:
        if len(eye_landmarks) != 6:
            logger.warning(f"Invalid eye landmarks count: {len(eye_landmarks)}")
            return 0.0
            
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if C == 0:
            logger.warning("Zero division in EAR calculation")
            return 0.0
            
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        logger.error(f"Error in eye_aspect_ratio: {str(e)}")
        return 0.0

def enhanced_eye_analysis(eye_landmarks):
    try:
        if len(eye_landmarks) != 6:
            logger.warning(f"Invalid eye landmarks count: {len(eye_landmarks)}")
            return {
                'ear': 0.0,
                'closure_ratio': 1.0,
                'pupil_position': (0, 0)
            }
            
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if C == 0:
            logger.warning("Zero division in enhanced eye analysis")
            return {
                'ear': 0.0,
                'closure_ratio': 1.0,
                'pupil_position': np.mean(eye_landmarks, axis=0) if eye_landmarks else (0, 0)
            }
            
        ear = (A + B) / (2.0 * C)
        
        return {
            'ear': ear,
            'closure_ratio': min(1.0, (C - (A+B)/2) / C),
            'pupil_position': np.mean(eye_landmarks, axis=0)
        }
    except Exception as e:
        logger.error(f"Error in enhanced_eye_analysis: {str(e)}")
        return {
            'ear': 0.0,
            'closure_ratio': 1.0,
            'pupil_position': (0, 0)
        }

def head_pose_estimation(landmarks, frame_shape):
    try:
        if len(landmarks) < max(HEAD_POSE_LANDMARKS):
            logger.warning("Not enough landmarks for head pose estimation")
            return np.zeros(3), np.zeros(3)
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])
        
        image_points = np.array([
            (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),
            (landmarks[199].x * frame_shape[1], landmarks[199].y * frame_shape[0]),
            (landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]),
            (landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]),
            (landmarks[61].x * frame_shape[1], landmarks[61].y * frame_shape[0]),
            (landmarks[291].x * frame_shape[1], landmarks[291].y * frame_shape[0])
        ], dtype="double")
        
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            logger.warning("Head pose estimation failed")
            return np.zeros(3), np.zeros(3)
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        return angles, translation_vector
    except Exception as e:
        logger.error(f"Error in head_pose_estimation: {str(e)}")
        return np.zeros(3), np.zeros(3)

def draw_enhanced_feedback(frame, detection_results, face_bbox=None):
    try:
        color_map = {
            "CALIBRATING": (255, 165, 0),
            "NORMAL": (0, 255, 0),
            "WARNING": (0, 255, 255),
            "ALERT": (0, 0, 255),
            "MONITORING": (255, 255, 0),
            "ERROR": (255, 0, 255)
        }
        
        color = color_map.get(detection_results['state'], (255, 255, 255))
        
        cv2.putText(frame, f"Status: {detection_results['state']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"Drowsiness: {detection_results['score']:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, 
                    f"Eyes: {detection_results['components']['eyes']:.2f} | "
                    f"Pose: {detection_results['components']['pose']:.2f} | "
                    f"Temp: {detection_results['components']['temporal']:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        cv2.putText(frame,
                    f"EAR: {detection_results['metrics']['ear']:.2f} (thresh: {detection_results['metrics']['ear_threshold']:.2f}) | "
                    f"Pitch: {detection_results['metrics']['pitch']:.1f}° | "
                    f"Blinks: {detection_results['metrics']['blink_rate']:.2f}/s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if detection_results['state'] == "CALIBRATING":
                progress = min(1.0, (time.time() - detector.calibration_start) / CALIBRATION_DURATION)
                cv2.rectangle(frame, (x1, y2 + 5), 
                              (int(x1 + (x2 - x1) * progress), y2 + 10),
                              (255, 165, 0), -1)
    except Exception as e:
        logger.error(f"Error in draw_enhanced_feedback: {str(e)}")

def initialize_models():
    try:
        logger.info("Initializing MediaPipe Face Mesh...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("Loading YOLOv8 face detection model...")
        model = YOLO('yolov8n-face.pt')
        
        logger.info("Initializing DrowsinessDetector...")
        detector = DrowsinessDetector()
        
        return face_mesh, model, detector
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

def main():
    try:
        face_mesh, model, detector = initialize_models()
        
        logger.info("Opening video capture...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video capture")
        
        frame_count = 0
        last_log_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                break
                
            frame_count += 1
            current_time = time.time()
            if current_time - last_log_time > 5:
                logger.info(f"Processing frame {frame_count}")
                last_log_time = current_time
            
            try:
                frame_h, frame_w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = model(frame, verbose=False)
                detection_results = None
                face_bbox = None
                
                for result in results:
                    for box in result.boxes:
                        if box.conf[0] < MIN_FACE_CONFIDENCE:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_bbox = (x1, y1, x2, y2)
                        
                        try:
                            face_results = face_mesh.process(rgb_frame[y1:y2, x1:x2])
                            
                            if face_results.multi_face_landmarks:
                                landmarks = face_results.multi_face_landmarks[0].landmark
                                
                                left_eye = []
                                right_eye = []
                                
                                for idx in LEFT_EYE_INDICES:
                                    landmark = landmarks[idx]
                                    px = int(landmark.x * frame_w)
                                    py = int(landmark.y * frame_h)
                                    left_eye.append((px, py))
                                
                                for idx in RIGHT_EYE_INDICES:
                                    landmark = landmarks[idx]
                                    px = int(landmark.x * frame_w)
                                    py = int(landmark.y * frame_h)
                                    right_eye.append((px, py))
                                
                                left_eye_metrics = enhanced_eye_analysis(left_eye)
                                right_eye_metrics = enhanced_eye_analysis(right_eye)
                                avg_ear = (left_eye_metrics['ear'] + right_eye_metrics['ear']) / 2
                                
                                pose_angles, _ = head_pose_estimation(landmarks, frame.shape)
                                
                                if detector.state == "CALIBRATING":
                                    if detector.calibrate(frame_count, CALIBRATION_DURATION):
                                        logger.info("Calibration complete!")
                                
                                detection_results = detector.detect(
                                    {
                                        'ear': avg_ear, 
                                        'closure_ratio': (left_eye_metrics['closure_ratio'] + right_eye_metrics['closure_ratio'])/2
                                    },
                                    pose_angles
                                )
                                
                                for point in left_eye + right_eye:
                                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
                        except Exception as e:
                            logger.error(f"Error processing face: {str(e)}")
                            continue
                
                if detection_results:
                    draw_enhanced_feedback(frame, detection_results, face_bbox)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.imshow("Enhanced Drowsiness Detection", frame)
                
                if detection_results and detection_results['state'] == "ALERT":
                    print("\aALERT! Drowsiness detected!")
                    
                if cv2.waitKey(1) == ord('q'):
                    logger.info("User requested quit")
                    break
            except Exception as e:
                logger.error(f"Error in main processing loop: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
    finally:
        logger.info("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        logger.info("Program exited")

if __name__ == "__main__":
    main()