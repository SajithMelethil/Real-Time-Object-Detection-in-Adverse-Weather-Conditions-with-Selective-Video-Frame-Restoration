from pathlib import Path
from dataclasses import dataclass
import torch
from typing import Tuple, Dict, List, Optional
import os

@dataclass
class PipelineConfig:
    """Configuration class for the weather processing pipeline"""
    
    # Directory configuration
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Processing directories
    RAW_FRAMES_DIR: Path = DATA_DIR / "processed/raw_frames"
    FILTERED_DIR: Path = DATA_DIR / "processed/filtered"
    RESTORED_DIR: Path = DATA_DIR / "processed/restored"
    DETECTED_DIR: Path = DATA_DIR / "processed/detected"
    OUTPUT_DIR: Path = DATA_DIR / "processed/output"
    FEEDBACK_DIR: Path = DATA_DIR / "feedback"
    LOGS_DIR: Path = DATA_DIR / "logs"
    
    # Model paths
    YOLOV8_MODEL: Path = MODELS_DIR / "yolov8n.pt"
    YOLOV11_MODEL: Path = MODELS_DIR / "yolo11x.pt"
    TRANSWEATHER_MODEL: Path = MODELS_DIR / "WeatherFormer_allweather.pth"
    WEATHER_CLASSIFIER_MODEL: Path = MODELS_DIR / "WeatherFormer_allweather.pth"
    
    # Processing parameters
    FRAME_EXTRACTION_FPS: int = 2
    MIN_VIDEO_FPS: int = 1
    DEFAULT_FPS: int = 30
    WEATHER_SEVERITY_THRESHOLD: float = 0.7
    CONTRAST_THRESHOLD: int = 30
    BLUR_THRESHOLD: int = 100
    
    # Detection parameters
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.4
    OBJECT_IOU_THRESHOLD: float = 0.5
    MAX_DETECTIONS_PER_FRAME: int = 100
    AGNOSTIC_NMS: bool = False
    OBJECT_INPUT_SIZE: Tuple[int, int] = (640, 640)
    
    # Restoration parameters
    RESTORATION_INPUT_SIZE: Tuple[int, int] = (256, 256)
    MAX_RESTORATION_BATCH_SIZE: int = 4
    
    # Visualization parameters
    BBOX_LINE_WIDTH: int = 1
    BBOX_FONT_SIZE: float = 0.5
    SHOW_LABELS: bool = True
    SHOW_CONFIDENCE: bool = True
    
    # Active learning
    ACTIVE_LEARNING_ENABLED: bool = True
    UNCERTAINTY_THRESHOLD: float = 0.3
    ENABLE_PRUNING: bool = True
    PRUNING_AMOUNT: float = 0.2
    
    # Performance
    HALF_PRECISION: bool = torch.cuda.is_available()
    MAX_PARALLEL_WORKERS: int = 4
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Initialize configuration with validation"""
        self._validate_paths()
        self._setup_directories()
        self._validate_parameters()
    
    def _validate_paths(self):
        """Verify critical model paths exist"""
        required_models = [
            (self.YOLOV11_MODEL, "YOLOv11 model"),
            (self.TRANSWEATHER_MODEL, "Weather restoration model"),
            (self.WEATHER_CLASSIFIER_MODEL, "Weather classifier model")
        ]
        
        for path, name in required_models:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found at {path}")
    
    def _setup_directories(self):
        """Create all required directories"""
        dirs = [
            self.RAW_FRAMES_DIR,
            self.FILTERED_DIR / "clear",
            self.FILTERED_DIR / "light",
            self.FILTERED_DIR / "severe",
            self.RESTORED_DIR,
            self.DETECTED_DIR,
            self.OUTPUT_DIR,
            self.FEEDBACK_DIR,
            self.LOGS_DIR
        ]
        
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create directory {d}: {str(e)}")
    
    def _validate_parameters(self):
        """Validate parameter ranges"""
        if not 0 < self.OBJECT_CONFIDENCE_THRESHOLD <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if not 0 < self.OBJECT_IOU_THRESHOLD <= 1:
            raise ValueError("IOU threshold must be between 0 and 1")
            
        if self.FRAME_EXTRACTION_FPS < self.MIN_VIDEO_FPS:
            raise ValueError(f"FPS cannot be below {self.MIN_VIDEO_FPS}")
    
    @property
    def model_config(self) -> Dict:
        """Get model-specific configuration"""
        return {
            'yolov11': {
                'conf': self.OBJECT_CONFIDENCE_THRESHOLD,
                'iou': self.OBJECT_IOU_THRESHOLD,
                'imgsz': self.OBJECT_INPUT_SIZE,
                'device': self.DEVICE
            },
            'transweather': {
                'imgsz': self.RESTORATION_INPUT_SIZE,
                'half': self.HALF_PRECISION
            }
        }
    
    @property
    def processing_config(self) -> Dict:
        """Get processing configuration"""
        return {
            'frame_skip': int(round(self.DEFAULT_FPS / self.FRAME_EXTRACTION_FPS)),
            'max_workers': self.MAX_PARALLEL_WORKERS,
            'active_learning': self.ACTIVE_LEARNING_ENABLED
        }