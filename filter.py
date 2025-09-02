import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Tuple, Any
from .config import PipelineConfig
from tqdm import tqdm
import random

class FrameFilter:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.yolov8 = YOLO(self.cfg.YOLOV8_MODEL)
        self._warmup_model()
    
    def _warmup_model(self):
        """Run dummy inference to initialize model"""
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.yolov8(dummy, verbose=False)
    
    def classify_frames(self, frame_paths: List[Path]) -> Tuple[Dict[str, List[Path]], Dict[str, Any]]:
        """Classify frames into weather/object categories with metrics"""
        categories = {
            "clear": [],
            "light": [],
            "severe": []
        }
        metrics = {
            'total_frames': len(frame_paths),
            'frames_processed': 0,
            'weather_assessments': 0,
            'clear_frames_kept': 0,
            'clear_frames_skipped': 0
        }
        
        # Create destination directories if they don't exist
        (self.cfg.FILTERED_DIR / "clear").mkdir(parents=True, exist_ok=True)
        (self.cfg.FILTERED_DIR / "light").mkdir(parents=True, exist_ok=True)
        (self.cfg.FILTERED_DIR / "severe").mkdir(parents=True, exist_ok=True)

        for frame_path in tqdm(frame_paths, desc="Classifying frames"):
            img = cv2.imread(str(frame_path))
            
            if img is None:
                print(f"Warning: Failed to read image {frame_path}. Skipping.")
                continue

            # Object detection
            results = self.yolov8(img, conf=self.cfg.OBJECT_CONFIDENCE_THRESHOLD, verbose=False)
            has_objects = len(results[0].boxes) > 0
            
            if not has_objects:
                # Only keep 1/14th of clear frames
                if random.random() < (1/14):
                    dest = self.cfg.FILTERED_DIR / "clear" / frame_path.name
                    categories["clear"].append(dest)
                    frame_path.rename(dest)
                    metrics['clear_frames_kept'] += 1
                else:
                    frame_path.unlink()  # Delete the frame we're not keeping
                    metrics['clear_frames_skipped'] += 1
            else:
                # Weather assessment - keep all non-clear frames
                is_severe = self._assess_weather(img)
                category = "severe" if is_severe else "light"
                dest = self.cfg.FILTERED_DIR / category / frame_path.name
                categories[category].append(dest)
                frame_path.rename(dest)
                metrics['weather_assessments'] += 1
            
            metrics['frames_processed'] += 1
        
        return categories, metrics
    
    def _assess_weather(self, img: np.ndarray) -> bool:
        """Determine if frame has severe weather degradation"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Contrast estimation
        contrast = np.std(gray)
        
        # Blur detection
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return (contrast < self.cfg.CONTRAST_THRESHOLD or 
                blur < self.cfg.BLUR_THRESHOLD)