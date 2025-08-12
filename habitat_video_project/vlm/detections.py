"""
Object detection results data structure
独立于VLFM的检测结果数据结构
"""

import json
from typing import List, Optional, Union

import numpy as np


class ObjectDetections:
    """Object detection results container"""
    
    def __init__(
        self,
        boxes: np.ndarray,
        logits: np.ndarray,
        phrases: List[str],
        image_source: Optional[np.ndarray] = None,
    ):
        """
        Initialize ObjectDetections
        
        Args:
            boxes: Normalized bounding boxes [x1, y1, x2, y2] in [0, 1]
            logits: Confidence scores
            phrases: Class names/phrases
            image_source: Source image for visualization
        """
        self.boxes = boxes
        self.logits = logits
        self.phrases = phrases
        self.image_source = image_source
        
        # Create annotated frame for visualization
        self.annotated_frame = self._create_annotated_frame()
    
    @property
    def num_detections(self) -> int:
        """Number of detections"""
        return len(self.logits)
    
    def filter_by_class(self, target_classes: List[str]) -> None:
        """
        Filter detections by class names
        
        Args:
            target_classes: List of target class names
        """
        if not target_classes:
            return
            
        # Convert to lowercase for case-insensitive matching
        target_classes_lower = [cls.lower() for cls in target_classes]
        
        # Find matching detections
        matching_indices = []
        for i, phrase in enumerate(self.phrases):
            phrase_lower = phrase.lower()
            if any(target_cls in phrase_lower for target_cls in target_classes_lower):
                matching_indices.append(i)
        
        # Filter results
        if matching_indices:
            self.boxes = self.boxes[matching_indices]
            self.logits = self.logits[matching_indices]
            self.phrases = [self.phrases[i] for i in matching_indices]
            # Recreate annotated frame
            self.annotated_frame = self._create_annotated_frame()
        else:
            # No matches found, clear all detections
            self.boxes = np.array([])
            self.logits = np.array([])
            self.phrases = []
            self.annotated_frame = self.image_source.copy() if self.image_source is not None else None
    
    def filter_by_conf(self, threshold: float) -> None:
        """
        Filter detections by confidence threshold
        
        Args:
            threshold: Confidence threshold
        """
        if self.num_detections == 0:
            return
            
        # Find detections above threshold
        valid_indices = np.where(self.logits >= threshold)[0]
        
        if len(valid_indices) > 0:
            self.boxes = self.boxes[valid_indices]
            self.logits = self.logits[valid_indices]
            self.phrases = [self.phrases[i] for i in valid_indices]
            # Recreate annotated frame
            self.annotated_frame = self._create_annotated_frame()
        else:
            # No detections above threshold, clear all
            self.boxes = np.array([])
            self.logits = np.array([])
            self.phrases = []
            self.annotated_frame = self.image_source.copy() if self.image_source is not None else None
    
    def _create_annotated_frame(self) -> Optional[np.ndarray]:
        """Create annotated frame for visualization"""
        if self.image_source is None or self.num_detections == 0:
            return self.image_source.copy() if self.image_source is not None else None
        
        annotated = self.image_source.copy()
        height, width = annotated.shape[:2]
        
        for i in range(self.num_detections):
            # Denormalize bounding box
            bbox = self.boxes[i] * np.array([width, height, width, height])
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{self.phrases[i]}: {self.logits[i]:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated
    
    def to_json(self) -> dict:
        """Convert to JSON-serializable format"""
        return {
            "boxes": self.boxes.tolist() if self.num_detections > 0 else [],
            "logits": self.logits.tolist() if self.num_detections > 0 else [],
            "phrases": self.phrases,
            "num_detections": self.num_detections
        }
    
    @classmethod
    def from_json(cls, data: dict, image_source: Optional[np.ndarray] = None) -> "ObjectDetections":
        """Create ObjectDetections from JSON data"""
        boxes = np.array(data.get("boxes", []))
        logits = np.array(data.get("logits", []))
        phrases = data.get("phrases", [])
        
        return cls(boxes, logits, phrases, image_source)
    
    def __repr__(self) -> str:
        return f"ObjectDetections(num_detections={self.num_detections})"


# Import cv2 for visualization
try:
    import cv2
except ImportError:
    print("Warning: cv2 not available, visualization will be disabled")
    cv2 = None 