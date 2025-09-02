import cv2
from pathlib import Path
from tqdm import tqdm
from .config import PipelineConfig
from typing import List, Tuple, Dict, Any

class FrameExtractor:
    def __init__(self, config: PipelineConfig):
        """
        Initialize the FrameExtractor with configuration.
        
        Args:
            config (PipelineConfig): Configuration object containing extraction parameters
        """
        self.cfg = config
        self.raw_frames_dir = self.cfg.DATA_DIR / "raw_frames"
        self.raw_frames_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Extract frames from video with adjusted frame skip (calculated skip - 4).
        
        Args:
            video_path (str): Path to the input video file
            
        Returns:
            Tuple[List[Path], Dict[str, Any]]: List of extracted frame paths and metrics dictionary
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"❌ Failed to open video: {video_path}")

        # Initialize metrics dictionary
        metrics = {
            'original_fps': cap.get(cv2.CAP_PROP_FPS),
            'frames_extracted': 0,
            'frame_skip': 0,
            'adjusted_skip': 0,
            'success': True,
            'video_duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }

        # Handle FPS calculation
        if metrics['original_fps'] <= 0 or metrics['original_fps'] < self.cfg.MIN_VIDEO_FPS:
            print(f"⚠️ Warning: Detected FPS ({metrics['original_fps']}) is invalid. Using default FPS: {self.cfg.DEFAULT_FPS}")
            metrics['original_fps'] = self.cfg.DEFAULT_FPS

        if self.cfg.FRAME_EXTRACTION_FPS <= 0:
            metrics['success'] = False
            raise ValueError("FRAME_EXTRACTION_FPS must be > 0")

        calculated_skip = int(metrics['original_fps'] / self.cfg.FRAME_EXTRACTION_FPS)
        metrics['frame_skip'] = calculated_skip
        metrics['adjusted_skip'] = max(1, calculated_skip - 4)  # Subtract 4 but don't go below 1
        


        frames = []
        try:
            with tqdm(desc="Extracting frames") as pbar:
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    
                    if frame_count % metrics['adjusted_skip'] == 0:
                        frame_path = self.raw_frames_dir / f"frame_{frame_count:06d}.jpg"
                        saved = cv2.imwrite(str(frame_path), frame)
                        if saved:
                            frames.append(frame_path)
                            metrics['frames_extracted'] += 1
                            pbar.update(1)
                        else:
                            print(f"❌ Failed to save frame at {frame_path}")

                    frame_count += 1

            metrics['success'] = len(frames) > 0
            metrics['actual_extraction_fps'] = metrics['original_fps'] / metrics['adjusted_skip']
            
            print("\nExtraction Summary:")
            print(f"- Frames extracted: {metrics['frames_extracted']}")
            print(f"✅ Extracted {len(frames)} frames to {self.raw_frames_dir}")
            
            return frames, metrics

        finally:
            cap.release()