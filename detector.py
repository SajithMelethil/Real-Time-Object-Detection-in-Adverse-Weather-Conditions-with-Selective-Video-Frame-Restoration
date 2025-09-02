from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .config import PipelineConfig
from tqdm import tqdm
import cv2
import numpy as np
from collections import defaultdict

class ObjectDetector:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.yolov11 = YOLO(self.cfg.YOLOV11_MODEL)
        self.class_names = self.yolov11.names  # Get class names from model
    
    def detect_objects(self, frame_paths: List[Path]) -> Tuple[List[Path], Dict[str, Any]]:
        """Run object detection on frames and return paths + metrics"""
        detected_paths = []
        detection_metrics = {
            'total_frames': len(frame_paths),
            'total_objects': 0,
            'avg_confidence': 0.0,
            'class_distribution': defaultdict(int),
            'confidences': [],
            'per_frame_stats': []
        }
        
        for frame_path in tqdm(frame_paths, desc="Detecting objects"):
            # Run detection
            results = self.yolov11(
                frame_path, 
                conf=self.cfg.OBJECT_CONFIDENCE_THRESHOLD,
                verbose=False
            )
            
            # Get detection data
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
            cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else np.array([])
            
            # Calculate frame-level metrics
            frame_objects = len(confs)
            frame_avg_conf = np.mean(confs).item() if frame_objects > 0 else 0.0
            
            # Update metrics
            detection_metrics['total_objects'] += frame_objects
            detection_metrics['confidences'].extend(confs.tolist())
            
            # Update class distribution
            for cls_id in cls_ids:
                class_name = self.class_names[int(cls_id)]
                detection_metrics['class_distribution'][class_name] += 1
            
            # Store per-frame stats
            detection_metrics['per_frame_stats'].append({
                'frame': frame_path.name,
                'objects': frame_objects,
                'avg_confidence': frame_avg_conf,
                'classes': [self.class_names[int(c)] for c in cls_ids]
            })
            
            # Generate and save annotated image
            annotated = results[0].plot()
            dest = self.cfg.DETECTED_DIR / frame_path.name
            cv2.imwrite(str(dest), annotated)
            detected_paths.append(dest)
        
        # Calculate aggregate metrics
        if detection_metrics['total_objects'] > 0:
            detection_metrics['avg_confidence'] = np.mean(detection_metrics['confidences']).item()
        
        # Convert defaultdict to regular dict
        detection_metrics['class_distribution'] = dict(detection_metrics['class_distribution'])
        
        return detected_paths, detection_metrics

    def get_detection_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a condensed summary of detection metrics"""
        return {
            'objects_detected': metrics['total_objects'],
            'avg_confidence': round(metrics['avg_confidence'], 3),
            'top_classes': dict(sorted(
                metrics['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])  # Top 5 classes
        }