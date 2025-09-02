import cv2
from pathlib import Path
from typing import List, Tuple, Dict
from .config import PipelineConfig
from tqdm import tqdm
import time
import logging
import numpy as np

class VideoMerger:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__)
    
    def merge_frames(self, frame_paths: List[Path]) -> Tuple[Path, Dict]:
        """
        Enhanced frame merging with validation and debugging
        
        Returns:
            Tuple: (output_path, metrics_dict)
        """
        metrics = {
            'start_time': time.time(),
            'valid_frames': 0,
            'invalid_frames': 0,
            'resolution': None,
            'fps': self.cfg.FRAME_EXTRACTION_FPS
        }
        
        try:
            # Validate input frames
            if not frame_paths:
                raise ValueError("Empty frame paths list")
            
            # Get video specs from first valid frame
            sample_frame, sample_path = self._get_valid_sample_frame(frame_paths)
            metrics['resolution'] = sample_frame.shape[:2]
            height, width = metrics['resolution']
            
            # Create output directory
            self.cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = self.cfg.OUTPUT_DIR / "output_video.mp4"
            
            # Initialize video writer with multiple codec attempts
            writer = self._init_video_writer(output_path, width, height)
            if writer is None:
                raise RuntimeError("All video writer initialization attempts failed")
            
            # Process all frames
            for frame_path in tqdm(sorted(frame_paths), desc="Merging frames"):
                frame = cv2.imread(str(frame_path))
                if frame is not None and frame.size > 0:
                    # Validate frame dimensions
                    if frame.shape[:2] == (height, width):
                        writer.write(frame)
                        metrics['valid_frames'] += 1
                    else:
                        self.logger.warning(f"Frame {frame_path.name} has incorrect dimensions")
                        metrics['invalid_frames'] += 1
                else:
                    metrics['invalid_frames'] += 1
            
            writer.release()
            
            # Final validation
            if not output_path.exists():
                raise RuntimeError("Output video file was not created")
            if metrics['valid_frames'] == 0:
                raise RuntimeError("No valid frames were written to video")
            
            metrics.update({
                'success': True,
                'duration_sec': time.time() - metrics['start_time'],
                'output_path': str(output_path)
            })
            
            return output_path, metrics
            
        except Exception as e:
            metrics.update({
                'success': False,
                'error': str(e),
                'duration_sec': time.time() - metrics['start_time']
            })
            self.logger.error(f"Video merging failed: {str(e)}")
            return None, metrics

    def _get_valid_sample_frame(self, frame_paths: List[Path]) -> Tuple[np.ndarray, Path]:
        """Find first valid frame to get video parameters"""
        for frame_path in frame_paths[:100]:  # Check first 100 frames max
            frame = cv2.imread(str(frame_path))
            if frame is not None and frame.size > 0:
                return frame, frame_path
        raise ValueError("No valid frames found in first 100 samples")

    def _init_video_writer(self, output_path: Path, width: int, height: int):
        """Initialize video writer with fallback codecs"""
        codecs = ['mp4v', 'avc1', 'X264', 'h264']
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.cfg.FRAME_EXTRACTION_FPS,
                    (width, height)
                )
                if writer.isOpened():
                    self.logger.info(f"Using video codec: {codec}")
                    return writer
            except Exception as e:
                self.logger.warning(f"Codec {codec} failed: {str(e)}")
        return None