import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ WARNING: ultralytics not available. Using fallback detection method.")

class CrowdDetector:
    """YOLOv8-based crowd detection system for real-time person counting"""
    
    def __init__(self, model_size: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the crowd detector with YOLOv8 or fallback method
        
        Args:
            model_size: YOLOv8 model size ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
            confidence_threshold: Minimum confidence for person detection
        """
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.model = None
        self.using_yolo = False
        
        # Try to initialize YOLO model
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_size)
                self.using_yolo = True
                print(f"✅ YOLOv8 model {model_size} loaded successfully")
            except Exception as e:
                print(f"❌ Error loading YOLOv8 model: {e}")
                # Fallback to nano model if specified model fails
                try:
                    self.model = YOLO('yolov8n.pt')
                    self.using_yolo = True
                    print("✅ Fallback to YOLOv8n model successful")
                except Exception as e2:
                    print(f"❌ Could not load any YOLOv8 model: {e2}")
                    self.using_yolo = False
        
        if not self.using_yolo:
            print("🔄 Using OpenCV HOG fallback detection method")
            # Initialize HOG descriptor for fallback
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        
        # COCO class ID for 'person' is 0
        self.person_class_id = 0
        
        # Performance tracking
        self.detection_history = []
        self.frame_count = 0
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in a single frame using YOLOv8 or HOG fallback
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of dictionaries containing detection information
        """
        try:
            if self.using_yolo:
                return self._detect_with_yolo(frame)
            else:
                return self._detect_with_hog(frame)
        except Exception as e:
            print(f"❌ Error in person detection: {e}")
            return []
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        """YOLOv8 detection method"""
        # Run YOLOv8 inference
        if self.model is not None:
            results = self.model(frame, verbose=False)
        else:
            return []
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                # Extract boxes, scores, and class IDs
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                # Filter for person detections with sufficient confidence
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    if class_id == self.person_class_id and score >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(score),
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
        
        # Update statistics
        self.frame_count += 1
        self.detection_history.append(len(detections))
        
        # Keep only recent history (last 100 frames)
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        return detections
    
    def _detect_with_hog(self, frame: np.ndarray) -> List[Dict]:
        """HOG fallback detection method"""
        # Run HOG detection
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)
        
        detections = []
        
        # Process detections
        for i, ((x, y, w, h), weight) in enumerate(zip(rects, weights)):
            if weight >= self.confidence_threshold:
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(weight),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'area': (x2 - x1) * (y2 - y1)
                }
                detections.append(detection)
        
        # Update statistics
        self.frame_count += 1
        self.detection_history.append(len(detections))
        
        # Keep only recent history (last 100 frames)
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        return detections
    
    def detect_people_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect people in multiple frames (batch processing)
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detection results for each frame
        """
        batch_results = []
        
        for frame in frames:
            detections = self.detect_people(frame)
            batch_results.append(detections)
        
        return batch_results
    
    def get_detection_stats(self) -> Dict:
        """
        Get statistics about recent detections
        
        Returns:
            Dictionary containing detection statistics
        """
        if not self.detection_history:
            return {
                'total_frames': 0,
                'avg_people_per_frame': 0,
                'max_people': 0,
                'min_people': 0,
                'recent_trend': 'stable'
            }
        
        recent_counts = self.detection_history[-10:]  # Last 10 frames
        
        stats = {
            'total_frames': self.frame_count,
            'avg_people_per_frame': np.mean(self.detection_history),
            'max_people': max(self.detection_history),
            'min_people': min(self.detection_history),
            'current_count': self.detection_history[-1] if self.detection_history else 0
        }
        
        # Determine trend
        if len(recent_counts) >= 3:
            recent_avg = np.mean(recent_counts[-3:])
            earlier_avg = np.mean(recent_counts[:-3]) if len(recent_counts) > 3 else recent_avg
            
            if recent_avg > earlier_avg * 1.1:
                stats['recent_trend'] = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                stats['recent_trend'] = 'decreasing'
            else:
                stats['recent_trend'] = 'stable'
        else:
            stats['recent_trend'] = 'stable'
        
        return stats
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict], 
                           show_confidence: bool = True, show_centers: bool = False) -> np.ndarray:
        """
        Draw detection results on the frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
            show_centers: Whether to show detection centers
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            center_x, center_y = detection['center']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            if show_confidence:
                label = f"Person: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            if show_centers:
                cv2.circle(annotated_frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        # Draw total count
        total_count = len(detections)
        count_text = f"Total People: {total_count}"
        cv2.putText(annotated_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        return annotated_frame
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for detections"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            print(f"✅ Confidence threshold updated to {new_threshold}")
        else:
            print(f"❌ Invalid confidence threshold: {new_threshold}. Must be between 0.0 and 1.0")
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_history = []
        self.frame_count = 0
        print("✅ Detection statistics reset")
