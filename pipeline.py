from pathlib import Path
from typing import List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import torch
import time
import logging
from .config import PipelineConfig
from .extractor import FrameExtractor
from .filter import FrameFilter
from .restorer import WeatherRestorer
from .detector import ObjectDetector
from .merger import VideoMerger

class WeatherPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.cfg = config if config else PipelineConfig()
        self._initialize_components()
        self._setup_parallel_execution()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_components(self):
        self.extractor = FrameExtractor(self.cfg)
        self.filter = FrameFilter(self.cfg)
        self.restorer = WeatherRestorer(self.cfg)
        self.detector = ObjectDetector(self.cfg)
        self.merger = VideoMerger(self.cfg)
        
    def _setup_parallel_execution(self):
        self.max_workers = getattr(self.cfg, 'MAX_PARALLEL_WORKERS', 2)
        self.executor = ThreadPoolExecutor(max_workers=min(4, self.max_workers))

    def run(self, video_path: Union[str, Path]) -> Optional[Path]:
        """Run the full pipeline and return output video path"""
        try:
            start_time = time.time()
            
            # 1. Frame Extraction
            raw_frames = self.extractor.extract_frames(str(video_path))
            if isinstance(raw_frames, tuple):
                raw_frames = raw_frames[0]  # Take just the frames if metrics are returned

            # 2. Frame Classification
            classified_frames = self.filter.classify_frames(raw_frames)
            if isinstance(classified_frames, tuple):
                classified_frames = classified_frames[0]  # Take just the classified frames

            # 3-4. Parallel Restoration and Detection
            restoration_task = self.executor.submit(
                self._process_severe_frames,
                classified_frames.get("severe", [])
            )
            detection_task = self.executor.submit(
                self._process_light_frames,
                classified_frames.get("light", [])
            )

            # Get results
            detected_restored = restoration_task.result()
            detected_light = detection_task.result()

            # 5. Video Compilation
            all_detected = detected_light + detected_restored
            output_path = self.merger.merge_frames(all_detected)
            if isinstance(output_path, tuple):
                output_path = output_path[0]  # Take just the path if metrics are returned

            self.logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds")
            return output_path

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return None
        finally:
            self.executor.shutdown(wait=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _process_light_frames(self, frames: List[Path]) -> List[Path]:
        """Process light weather frames (just detection)"""
        if not frames:
            return []
        result = self.detector.detect_objects(frames)
        return result[0] if isinstance(result, tuple) else result

    def _process_severe_frames(self, frames: List[Path]) -> List[Path]:
        """Process severe weather frames (restoration + detection)"""
        if not frames:
            return []
            
        # Restore frames
        restored_frames = self.restorer.restore_frames(frames)
        if isinstance(restored_frames, tuple):
            restored_frames = restored_frames[0]
            
        # Detect objects
        detected_frames = self.detector.detect_objects(restored_frames)
        return detected_frames[0] if isinstance(detected_frames, tuple) else detected_frames