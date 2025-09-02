import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from .config import PipelineConfig
from tqdm import tqdm
from .WeatherFormer import WeatherFormer
import logging

class WeatherRestorer:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.device = self.cfg.DEVICE
        self.model = self._load_optimized_model()
        self.batch_size = self._determine_optimal_batch_size()  # New dynamic batch sizing
        
    def _determine_optimal_batch_size(self) -> int:
        """Automatically determine optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return 1
            
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_mem > 16:  # High-end GPU
            return 8
        elif total_mem > 8:
            return 4
        return 2

    def _load_optimized_model(self) -> nn.Module:
        """Load model with quantization and pruning"""
        model = WeatherFormer()
        
        # Load weights
        state_dict = torch.load(str(self.cfg.TRANSWEATHER_MODEL), map_location=self.device)
        model.load_state_dict(state_dict)
        
        # Apply optimizations
        if self.cfg.HALF_PRECISION:
            model = model.half()
            
        # Prune small weights
        if hasattr(torch.nn.utils, 'prune'):
            parameters_to_prune = [
                (module, 'weight') for module in model.modules() 
                if isinstance(module, nn.Conv2d)
            ]
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=0.2  # Prune 20% of smallest weights
            )
            
        model.eval()
        return model.to(self.device)

    def restore_frames(self, frame_paths: List[Path]) -> List[Path]:
        """Optimized restoration with batch processing"""
        # Group frames into optimal batches
        batches = [frame_paths[i:i + self.batch_size] 
                  for i in range(0, len(frame_paths), self.batch_size)]
        
        restored_paths = []
        for batch in tqdm(batches, desc="Restoring frames"):
            try:
                # Process batch on GPU
                with torch.cuda.amp.autocast(enabled=self.cfg.HALF_PRECISION):
                    restored_batch = self._process_batch(batch)
                restored_paths.extend(restored_batch)
                
                # Clear cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                # Handle OOM by reducing batch size
                self.batch_size = max(1, self.batch_size // 2)
                return self.restore_frames(frame_paths)  # Retry with smaller batches
                
        return restored_paths

    def _process_batch(self, batch_paths: List[Path]) -> List[Path]:
        """Process batch with improved model output handling"""
        batch_tensors = []
        original_shapes = []
        valid_paths = []

        # Load and validate frames
        for frame_path in batch_paths:
            try:
                img = cv2.imread(str(frame_path))
                if img is None:
                    raise ValueError(f"Failed to read image {frame_path}")

                original_shapes.append(img.shape[:2])
                tensor = self._preprocess(img)
                batch_tensors.append(tensor)
                valid_paths.append(frame_path)
            except Exception as e:
                self.logger.error(f"Error loading frame {frame_path}: {str(e)}")
                continue

        if not batch_tensors:
            return []

        try:
            # Process batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            if self.cfg.HALF_PRECISION and torch.cuda.is_available():
                batch_tensor = batch_tensor.half()
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=self.cfg.HALF_PRECISION):
                        restored_batch = self.model(batch_tensor)
                else:
                    restored_batch = self.model(batch_tensor)

                # If model returns a tuple (common in some architectures)
                if isinstance(restored_batch, tuple):
                    restored_batch = restored_batch[0]  # Take first output

                # Ensure we have a 4D tensor [B, C, H, W]
                if restored_batch.dim() == 3:
                    restored_batch = restored_batch.unsqueeze(0)

        except Exception as e:
            self.logger.error(f"Error during model inference: {str(e)}")
            return []

        # Save results
        results = []
        for i, frame_path in enumerate(valid_paths):
            try:
                output = self._postprocess(restored_batch[i], original_shapes[i])
                dest = self.cfg.RESTORED_DIR / frame_path.name
                cv2.imwrite(str(dest), output)
                results.append(dest)
            except Exception as e:
                self.logger.error(f"Error saving frame {frame_path}: {str(e)}")
                continue

        return results

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Optimized preprocessing with validation and resizing"""
        try:
            # Validate input
            if not isinstance(img, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(img)}")
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Invalid image shape {img.shape}, expected HxWx3")
            
            # Resize to multiple of 32
            h, w = img.shape[:2]
            new_h = (h // 32) * 32
            new_w = (w // 32) * 32
            
            if new_h != h or new_w != w:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to tensor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def _postprocess(self, tensor: torch.Tensor, original_shape: tuple) -> np.ndarray:
        """Convert tensor back to image with robust error handling"""
        try:
            # Ensure tensor is float32 and on CPU
            tensor = tensor.float().cpu()

            # Handle different tensor shapes:
            if tensor.dim() == 4:  # [B, C, H, W]
                tensor = tensor[0]  # Take first image in batch

            # Check if this is a feature map vs output image
            if tensor.size(0) > 4:  # Likely a feature map
                # Convert feature map to image (simple visualization)
                # Take mean across channels and normalize
                tensor = tensor.mean(dim=0, keepdim=True)
                tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

            # Handle valid image tensors
            if tensor.dim() == 3 and tensor.size(0) in [1, 3]:
                if tensor.size(0) == 1:  # Grayscale
                    img = tensor.squeeze().numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:  # RGB
                    img = tensor.permute(1, 2, 0).numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Resize back to original if needed
                if img.shape[:2] != original_shape:
                    img = cv2.resize(img, (original_shape[1], original_shape[0]))
                return img

            raise ValueError(f"Unsupported tensor shape {tensor.shape}")

        except Exception as e:
            self.logger.error(f"Postprocessing error: {str(e)}")
            # Return black image of correct size as fallback
            return np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)