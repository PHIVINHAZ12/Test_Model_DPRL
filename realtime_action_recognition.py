#!/usr/bin/env python3
"""
Real-time action recognition using trained DPRL model
"""

import sys
import os
import cv2
import torch
import numpy as np
import time
from collections import deque
import mediapipe as mp
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import YOLO for person detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Will process full frame for pose detection.")

# Add DPRL_Minimal to path
sys.path.append(r'D:\CODE_FOR_VINH\CODE\NTUActionLearning\NTUActionLearning\DPRL_Minimal')

# Import DPRL components
from config import Config
from models.gcnn import GCNN
from models.fdnet_paper_compliant import PaperCompliantFDNet

class DPRLActionRecognizer:
    """Real-time action recognition using DPRL model"""
    
    def __init__(self, model_path: str = 'DPRL_Minimal/outputs'):
        """
        Initialize DPRL action recognizer
        
        Args:
            model_path: Path to trained model checkpoints
        """
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.gcnn = GCNN(self.config).to(self.device)
        self.fdnet = PaperCompliantFDNet(self.config).to(self.device)
        self.classifier = torch.nn.Linear(self.config.gcnn_output_dim, self.config.num_classes).to(self.device)
        
        # Load trained weights
        self.load_models(model_path)
        
        # Set to evaluation mode
        self.gcnn.eval()
        self.fdnet.eval()
        self.classifier.eval()
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # YOLO setup for person detection
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                if self.device.type == 'cuda':
                    self.yolo_model.to(self.device)
                print("✅ YOLO person detection enabled")
            except Exception as e:
                print(f"⚠️ YOLO initialization failed: {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=self.config.max_frames)
        self.min_frames_for_prediction = 16
        
        print(f"✅ DPRL Action Recognizer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Actions: {self.config.action_classes}")
        print(f"   - Buffer size: {self.config.max_frames}")
    
    def load_models(self, model_path: str):
        """Load trained model weights"""
        try:
            # Try to load from training script output paths
            possible_paths = [
                # Step 3 models (best performance after mutual boosting)
                os.path.join(model_path, 'step3_gcnn_boosted.pth'),
                os.path.join(model_path, 'step3_fdnet_boosted.pth'),
                # Step 1 model (GCNN baseline)
                os.path.join(model_path, 'step1_paper_compliant_gcnn.pth'),
                # Step 2 model (FDNet)
                os.path.join(model_path, 'step2_paper_compliant_fdnet.pth'),
                # Fallback to root directory
                'step3_gcnn_boosted.pth',
                'step3_fdnet_boosted.pth',
                'step1_gcnn_for_reward.pth',
                'step2_paper_compliant_fdnet.pth'
            ]
            
            # Load models from different files (as saved by training script)
            models_loaded = []
            
            # Try to load GCNN + Classifier (from Step 1 or Step 3)
            gcnn_paths = [
                os.path.join(model_path, 'step3_gcnn_boosted.pth'),
                os.path.join(model_path, 'step1_gcnn_for_reward.pth'),
                'step3_gcnn_boosted.pth',
                'step1_gcnn_for_reward.pth'
            ]
            
            for path in gcnn_paths:
                if os.path.exists(path):
                    checkpoint = torch.load(path, map_location=self.device)
                    if 'gcnn_state_dict' in checkpoint:
                        self.gcnn.load_state_dict(checkpoint['gcnn_state_dict'])
                        models_loaded.append(f"GCNN from {path}")
                    if 'classifier_state_dict' in checkpoint:
                        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                        models_loaded.append(f"Classifier from {path}")
                    break
            
            # Try to load FDNet (from Step 2 or Step 3)
            fdnet_paths = [
                os.path.join(model_path, 'step3_fdnet_boosted.pth'),
                os.path.join(model_path, 'step2_paper_compliant_fdnet.pth'),
                'step3_fdnet_boosted.pth',
                'step2_paper_compliant_fdnet.pth'
            ]
            
            for path in fdnet_paths:
                if os.path.exists(path):
                    checkpoint = torch.load(path, map_location=self.device)
                    if 'fdnet_state_dict' in checkpoint:
                        self.fdnet.load_state_dict(checkpoint['fdnet_state_dict'])
                        models_loaded.append(f"FDNet from {path}")
                    break
            
            if not models_loaded:
                print("⚠️  No trained model found. Using random weights.")
                print("   Run training first:")
                print("   cd DPRL_Minimal && python train_dprl_paper_compliant.py")
                return
            
            print("✅ Models loaded:")
            for model in models_loaded:
                print(f"   - {model}")
                
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Using random weights. Run training first.")
    
    def extract_skeleton_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract skeleton from frame using MediaPipe
        
        Args:
            frame: Input frame (H, W, 3)
            
        Returns:
            Skeleton data (25, 3) or None if no pose detected
        """
        try:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose_detector.process(frame_rgb)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = results.pose_landmarks.landmark
            if len(landmarks) < 33:
                return None
            
            # Convert MediaPipe 33 landmarks to NTU RGB+D 25 joints
            skeleton = np.zeros((25, 3))
            
            # Calculate spine joints (MediaPipe doesn't have direct spine points)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate spine points from shoulder and hip midpoints
            shoulder_mid = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2,
                (left_shoulder.z + right_shoulder.z) / 2
            ])
            
            hip_mid = np.array([
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2,
                (left_hip.z + right_hip.z) / 2
            ])
            
            # NTU RGB+D joint mapping (0-based indexing)
            # 0: base of spine, 1: middle of spine, 2: neck, 3: head
            skeleton[0] = hip_mid  # base of spine
            skeleton[1] = (shoulder_mid + hip_mid) / 2  # middle of spine
            skeleton[20] = shoulder_mid  # spine shoulder
            skeleton[2] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]  # neck (nose approximation)
            skeleton[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]  # head (nose)
            
            # Left arm: 4-left shoulder, 5-left elbow, 6-left wrist, 7-left hand
            skeleton[4] = [landmarks[11].x, landmarks[11].y, landmarks[11].z]  # left shoulder
            skeleton[5] = [landmarks[13].x, landmarks[13].y, landmarks[13].z]  # left elbow
            skeleton[6] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]  # left wrist
            skeleton[7] = [landmarks[19].x, landmarks[19].y, landmarks[19].z]  # left hand (index)
            
            # Right arm: 8-right shoulder, 9-right elbow, 10-right wrist, 11-right hand
            skeleton[8] = [landmarks[12].x, landmarks[12].y, landmarks[12].z]  # right shoulder
            skeleton[9] = [landmarks[14].x, landmarks[14].y, landmarks[14].z]  # right elbow
            skeleton[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right wrist
            skeleton[11] = [landmarks[20].x, landmarks[20].y, landmarks[20].z]  # right hand (index)
            
            # Left leg: 12-left hip, 13-left knee, 14-left ankle, 15-left foot
            skeleton[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z]  # left hip
            skeleton[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z]  # left knee
            skeleton[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z]  # left ankle
            skeleton[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z]  # left foot
            
            # Right leg: 16-right hip, 17-right knee, 18-right ankle, 19-right foot
            skeleton[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z]  # right hip
            skeleton[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z]  # right knee
            skeleton[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z]  # right ankle
            skeleton[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z]  # right foot
            
            # Hand tips and thumbs
            skeleton[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z]  # left hand tip (pinky)
            skeleton[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z]  # left thumb
            skeleton[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z]  # right hand tip (pinky)
            skeleton[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z]  # right thumb
            
            return skeleton.astype(np.float32)  # (25, 3)
            
        except Exception as e:
            print(f"Error extracting skeleton: {e}")
            return None
    
    def preprocess_skeleton_sequence(self, skeleton_sequence: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess skeleton sequence for DPRL model
        
        Args:
            skeleton_sequence: List of skeleton frames [(25, 3), ...]
            
        Returns:
            Preprocessed tensor (1, T, 25, 3)
        """
        # Stack frames
        sequence = np.stack(skeleton_sequence, axis=0)  # (T, 25, 3)
        
        # Normalize skeleton (center around hip)
        if sequence.shape[0] > 0:
            hip_center = sequence[:, 0:1, :].copy()  # (T, 1, 3) - base of spine
            sequence = sequence - hip_center
            
            # Scale normalization
            scale = np.std(sequence) + 1e-6
            sequence = sequence / scale
        
        # Pad or truncate to max_frames
        target_frames = self.config.max_frames
        if len(sequence) < target_frames:
            # Pad with last frame
            last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, 25, 3))
            pad_count = target_frames - len(sequence)
            padding = np.repeat(last_frame, pad_count, axis=0)
            sequence = np.concatenate([sequence, padding], axis=0)
        else:
            # Truncate
            sequence = sequence[:target_frames]
        
        # Convert to tensor
        tensor = torch.from_numpy(sequence).float().unsqueeze(0)  # (1, T, 25, 3)
        return tensor.to(self.device)
    
    def predict_action(self, skeleton_sequence: List[np.ndarray]) -> Tuple[str, float]:
        """
        Predict action from skeleton sequence
        
        Args:
            skeleton_sequence: List of skeleton frames
            
        Returns:
            (action_name, confidence)
        """
        if len(skeleton_sequence) < self.min_frames_for_prediction:
            return "Waiting for more frames...", 0.0
        
        try:
            # Preprocess
            input_tensor = self.preprocess_skeleton_sequence(skeleton_sequence)
            
            with torch.no_grad():
                # Step 1: Use FDNet to select important frames
                initial_frames = self.fdnet.uniform_sample(1, self.config.max_frames)
                
                # Progressive frame selection (simplified for real-time)
                selected_frames = self.fdnet.progressive_adjustment(
                    input_tensor, initial_frames, 
                    self.gcnn, self.classifier, 
                    torch.tensor([0]).to(self.device)  # dummy target
                )[0]  # Get final frames
                
                # Step 2: Extract selected frames
                selected_input = self.fdnet.extract_selected_frames(input_tensor, selected_frames)
                
                # Step 3: Process with GCNN
                gcnn_features = self.gcnn(selected_input)
                
                # Step 4: Classify
                logits = self.classifier(gcnn_features)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get prediction
                pred_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, pred_idx].item()
                
                action_name = self.config.action_classes[pred_idx]
                
                return action_name, confidence
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Process a single frame and return prediction
        
        Args:
            frame: Input frame (H, W, 3)
            
        Returns:
            (action_name, confidence, skeleton_points, person_box)
        """
        person_box = None
        
        # Use YOLO to detect person if available
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(frame)[0]
                
                # Find person with highest confidence
                best_conf = 0
                best_box = None
                
                for box in results.boxes.data:
                    cls_id = int(box[5].item())
                    conf = box[4].item()
                    
                    if cls_id == 0 and conf > best_conf:  # class 0 is person
                        best_conf = conf
                        x1, y1, x2, y2 = map(int, box[:4])
                        best_box = (x1, y1, x2, y2)
                
                if best_box is not None:
                    person_box = best_box
                    x1, y1, x2, y2 = best_box
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Extract skeleton from person crop
                    skeleton = self.extract_skeleton_from_frame(person_crop)
                    
                    if skeleton is not None:
                        # Adjust skeleton coordinates back to full frame
                        skeleton[:, 0] = skeleton[:, 0] * (x2 - x1) / frame.shape[1] + x1 / frame.shape[1]
                        skeleton[:, 1] = skeleton[:, 1] * (y2 - y1) / frame.shape[0] + y1 / frame.shape[0]
                else:
                    skeleton = None
            except Exception as e:
                print(f"YOLO detection error: {e}")
                skeleton = self.extract_skeleton_from_frame(frame)
        else:
            # Fallback to full frame processing
            skeleton = self.extract_skeleton_from_frame(frame)
        
        if skeleton is None:
            return "No person detected", 0.0, None, person_box
        
        # Add to buffer
        self.frame_buffer.append(skeleton)
        
        # Predict action
        action_name, confidence = self.predict_action(list(self.frame_buffer))
        
        # Convert skeleton to image coordinates for visualization
        h, w = frame.shape[:2]
        skeleton_points = skeleton.copy()
        skeleton_points[:, 0] *= w  # x coordinates
        skeleton_points[:, 1] *= h  # y coordinates
        
        return action_name, confidence, skeleton_points, person_box
    
    def draw_skeleton(self, frame: np.ndarray, skeleton_points: np.ndarray):
        """Draw skeleton on frame"""
        if skeleton_points is None:
            return
        
        # NTU RGB+D skeleton connections (dataset standard)
        connections = [
            (0, 1), (1, 20), (20, 2), (2, 3),               # spine to head
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (10, 23),  # right arm
            (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (6, 21),       # left arm
            (0, 16), (16, 17), (17, 18),                   # left leg
            (0, 12), (12, 13), (13, 14),                   # right leg
            (14, 15), (18, 19)                   # feet
        ]
        
        # Draw joints
        for i, point in enumerate(skeleton_points):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        for start, end in connections:
            if start < len(skeleton_points) and end < len(skeleton_points):
                x1, y1 = int(skeleton_points[start][0]), int(skeleton_points[start][1])
                x2, y2 = int(skeleton_points[end][0]), int(skeleton_points[end][1])
                
                if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                    0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

def main():
    """Main function for real-time action recognition"""
    # Initialize recognizer
    recognizer = DPRLActionRecognizer()
    
    # Camera configuration (restore original URLs)
    CAMERA_URLS = [
        #('IP Camera', 'http://61.214.197.204:1025/-wvhttp-01-/GetOneShot?image_size=1080x720&frame_count=1000000000&quality=100&compression=0&rotation=0&flip=0&mirror=0&timestamp=1'),
        ('RTSP Camera', 'rtsp://admin:L286E585@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1')
        #('YouTube Stream', 'https://www.youtube.com/shorts/81pj4qHZOq8?feature=share'),
        #('Webcam', 0)  # Fallback
    ]
    
    # Try camera URLs in order with timeout
    cap = None
    successful_camera = None
    
    for i, (name, url) in enumerate(CAMERA_URLS):
        print(f"🔍 Trying {name}: {url}")
        
        try:
            cap = cv2.VideoCapture(url)
            
            # Set timeout for network cameras
            if isinstance(url, str):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)  # 3 second timeout
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Successfully connected to {name}")
                    successful_camera = name
                    break
                else:
                    print(f"❌ {name} opened but cannot read frames")
                    cap.release()
                    cap = None
            else:
                print(f"❌ Failed to connect to {name}")
                cap = None
                
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            cap = None
    
    if cap is None or not cap.isOpened():
        print("❌ Cannot open any camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎥 Starting real-time action recognition...")
    print("   Press 'q' to quit")
    print("   Press 'r' to reset buffer")
    
    # FPS calculation variables
    fps_counter = 0
    fps_time = time.time()
    fps = 0.0
    frame_times = []
    
    while True:
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        action_name, confidence, skeleton_points, person_box = recognizer.process_frame(frame)
        
        # Draw person bounding box if available
        if person_box is not None:
            x1, y1, x2, y2 = person_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 139, 34), 2)
            cv2.putText(frame, f"{action_name} ({confidence*100:.1f}%)", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 139, 34), 2)
        
        # Draw skeleton
        recognizer.draw_skeleton(frame, skeleton_points)
        
        # Draw information
        cv2.putText(frame, f"Action: {action_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 139, 34), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 139, 34), 2)
        cv2.putText(frame, f"Buffer: {len(recognizer.frame_buffer)}/{recognizer.config.max_frames}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 139, 34), 2)
        
        # Calculate real-time FPS using moving average
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        frame_times.append(frame_duration)
        
        # Keep only last 30 frame times for moving average
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        # Calculate average FPS
        if len(frame_times) > 0:
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # Display FPS continuously with color coding
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Add processing time info
        cv2.putText(frame, f"Frame Time: {frame_duration*1000:.1f}ms", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 139, 34), 2)
        
        # Show frame
        cv2.imshow('DPRL Action Recognition - IP Camera', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.frame_buffer.clear()
            print("Buffer reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Real-time recognition stopped")

if __name__ == "__main__":
    main()