"""
Simplified matching service with basic image comparison.
Uses simple computer vision methods for actual image matching.
"""

import logging
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional
from backend.models.tile import Tile

# Advanced preprocessing dependencies
try:
    from skimage import exposure, restoration, color
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ViT dependencies (with fallback if not available)
try:
    import torch
    from transformers import ViTFeatureExtractor, ViTModel
    import time
    import warnings
    from PIL import ImageEnhance
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class MatchingConfig:
    """Configuration class for matching service parameters."""
    
    # ViT Model Configuration
    VIT_MODEL_NAME = "google/vit-base-patch16-224"
    VIT_SCALES = [224, 256, 192]
    VIT_BASE_DIM = 768
    VIT_LARGE_DIM = 1024
    VIT_MAX_LOAD_ATTEMPTS = 3
    
    # Image Processing Configuration
    MAX_IMAGE_SIZE = 1024
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    CONTRAST_ENHANCEMENT_FACTOR = 1.1
    BILATERAL_FILTER_D = 9
    BILATERAL_FILTER_SIGMA = 75
    
    # Similarity Calculation Weights
    ENSEMBLE_WEIGHTS = {
        'hist': 0.1,
        'orb': 0.2, 
        'enhanced_vit': 0.7
    }
    
    ENHANCED_ENSEMBLE_WEIGHTS = {
        'hist': 0.05,
        'orb': 0.1,
        'enhanced_single': 0.15,
        'enhanced_multi_layer': 0.3,
        'enhanced_multi_scale': 0.4
    }
    
    ADVANCED_PREPROCESSING_WEIGHTS = {
        'hist': 0.33,
        'orb': 0.33,
        'vit': 0.34
    }
    
    VIT_SIMILARITY_WEIGHTS = {
        'cosine': 0.4,
        'euclidean': 0.3,
        'correlation': 0.2,
        'manhattan': 0.1
    }
    
    # Performance Configuration
    DEFAULT_DB_LIMIT = 50
    FEATURE_CACHE_SIZE = 100
    
    @classmethod
    def validate_weights(cls, weights_dict: dict) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total = sum(weights_dict.values())
        return abs(total - 1.0) < 0.01

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Advanced image preprocessing for improved matching accuracy."""
    
    def __init__(self):
        """Initialize the image preprocessor."""
        self.clahe = cv2.createCLAHE(
            clipLimit=MatchingConfig.CLAHE_CLIP_LIMIT, 
            tileGridSize=MatchingConfig.CLAHE_TILE_GRID_SIZE
        )
    
    def preprocess(self, image, enable_tile_extraction=False):
        """
        Apply all preprocessing steps to an image.
        
        Args:
            image: Input BGR image (OpenCV format)
            enable_tile_extraction: Whether to extract tile from background
            
        Returns:
            Preprocessed image
        """
        if image is None:
            raise ValueError("Invalid image provided")
            
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Apply individual preprocessing steps
        processed = self.normalize_lighting(processed)
        processed = self.enhance_contrast(processed)
        processed = self.remove_noise(processed)
        
        if enable_tile_extraction:
            processed = self.extract_tile(processed)
        
        return processed
    
    def normalize_lighting(self, image):
        """
        Normalize lighting conditions and reduce shadows.
        
        Args:
            image: Input BGR image
            
        Returns:
            Image with normalized lighting
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split into channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def enhance_contrast(self, image):
        """
        Enhance image contrast to make features more distinguishable.
        
        Args:
            image: Input BGR image
            
        Returns:
            Contrast-enhanced image
        """
        if not SKIMAGE_AVAILABLE:
            # Fallback to OpenCV histogram equalization
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Perform contrast stretching on V channel
        v_eq = exposure.equalize_hist(v)
        v_eq = (v_eq * 255).astype(np.uint8)
        
        # Merge channels and convert back to BGR
        hsv_eq = cv2.merge([h, s, v_eq])
        enhanced = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def remove_noise(self, image):
        """
        Remove noise from the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Denoised image
        """
        # Validate image size to prevent DoS attacks
        height, width = image.shape[:2]
        if height * width > MatchingConfig.MAX_IMAGE_SIZE * MatchingConfig.MAX_IMAGE_SIZE:
            # Resize large images before processing
            scale = MatchingConfig.MAX_IMAGE_SIZE / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(
            image, 
            MatchingConfig.BILATERAL_FILTER_D, 
            MatchingConfig.BILATERAL_FILTER_SIGMA, 
            MatchingConfig.BILATERAL_FILTER_SIGMA
        )
        
        return denoised
    
    def extract_tile(self, image):
        """
        Attempt to extract the tile from the background.
        
        Args:
            image: Input BGR image
            
        Returns:
            Image with isolated tile 
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, threshold = cv2.threshold(blurred, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (assumed to be the tile)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the largest contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply the mask to the original image
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # Set background to white
        
        return result

class FeatureCache:
    """Simple LRU cache for feature vectors."""
    
    def __init__(self, max_size: int = MatchingConfig.FEATURE_CACHE_SIZE):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached features by key."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key].copy()
        return None
    
    def put(self, key: str, features: np.ndarray) -> None:
        """Cache features with LRU eviction."""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = features.copy()
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()

class EnhancedViTFeatureExtractor:
    """Enhanced Vision Transformer feature extractor with multi-layer and multi-scale capabilities"""
    
    def __init__(self, model_name: str = None, 
                 use_multi_layer: bool = True, 
                 use_multi_scale: bool = True,
                 scales: list = None):
        self.model_name = model_name or MatchingConfig.VIT_MODEL_NAME
        self.use_multi_layer = use_multi_layer
        self.use_multi_scale = use_multi_scale
        self.scales = scales or MatchingConfig.VIT_SCALES
        self._model = None
        self._extractor = None
        self._device = None
        self._available = None
        self._load_attempts = 0
        self._max_load_attempts = MatchingConfig.VIT_MAX_LOAD_ATTEMPTS
        self.logger = logging.getLogger(self.__class__.__name__)
        self._feature_cache = FeatureCache()
    
    def _check_availability(self) -> bool:
        """Check if VIT dependencies are available"""
        if self._available is None:
            self._available = VIT_AVAILABLE
            if self._available:
                self.logger.info("Enhanced VIT dependencies are available")
            else:
                self.logger.warning("Enhanced VIT dependencies not available")
        return self._available
    
    def _load_model(self):
        """Load VIT model with improved error handling and multi-layer support"""
        if not self._check_availability():
            raise ImportError("Enhanced VIT dependencies not available")
        
        if self._model is not None and self._extractor is not None:
            return
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
                # Enable output_hidden_states for multi-layer extraction
                self._model = ViTModel.from_pretrained(
                    self.model_name, 
                    output_hidden_states=self.use_multi_layer
                ).to(self._device)
                self._model.eval()
            
            self.logger.info(f"Enhanced VIT model loaded on {self._device} with multi-layer={self.use_multi_layer}, multi-scale={self.use_multi_scale}")
            
        except Exception as e:
            self._load_attempts += 1
            if self._load_attempts < self._max_load_attempts:
                self.logger.warning(f"Enhanced VIT model loading failed (attempt {self._load_attempts}): {e}")
                time.sleep(1)
                return self._load_model()
            else:
                raise RuntimeError(f"Failed to load Enhanced VIT model after {self._max_load_attempts} attempts: {e}")
    
    def extract_features(self, image_bytes: bytes) -> np.ndarray:
        """Extract enhanced VIT features with multi-layer and multi-scale processing"""
        try:
            # Generate cache key from image bytes hash
            import hashlib
            cache_key = f"{hashlib.md5(image_bytes).hexdigest()}_{self.use_multi_layer}_{self.use_multi_scale}"
            
            # Check cache first
            cached_features = self._feature_cache.get(cache_key)
            if cached_features is not None:
                self.logger.debug("Using cached features")
                return cached_features
            
            self._load_model()
            
            # Convert to PIL Image with better error handling
            pil_image = self._bytes_to_pil_image(image_bytes)
            if pil_image is None:
                raise ValueError("Failed to decode image")
            
            # Preprocess image
            pil_image = self._preprocess_image(pil_image)
            
            if self.use_multi_scale:
                # Multi-scale feature extraction
                features = self._extract_multiscale_features(pil_image)
            else:
                # Single-scale feature extraction
                features = self._extract_single_scale_features(pil_image)
            
            # Safe GPU memory cleanup
            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            features = features.astype(np.float32)
            
            # Cache the features
            self._feature_cache.put(cache_key, features)
            
            return features
                
        except Exception as e:
            self.logger.error(f"Enhanced VIT feature extraction failed: {e}")
            # Return consistent fallback dimensions
            fallback_dim = self.get_feature_dimension()
            return np.zeros(fallback_dim, dtype=np.float32)
    
    def _bytes_to_pil_image(self, image_bytes: bytes):
        """Convert bytes to PIL Image with enhanced error handling"""
        try:
            img_stream = BytesIO(image_bytes)
            pil_img = Image.open(img_stream)
            
            # Handle different image modes
            if pil_img.mode in ['RGBA', 'LA']:
                # Create white background for transparency
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                if pil_img.mode == 'RGBA':
                    background.paste(pil_img, mask=pil_img.split()[-1])
                else:
                    background.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = background
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            return pil_img
            
        except Exception as e:
            self.logger.error(f"Failed to convert bytes to PIL image: {e}")
            return None
    
    def _preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """Preprocess image for better feature extraction"""
        # Resize very large images to prevent memory issues
        if max(pil_image.size) > MatchingConfig.MAX_IMAGE_SIZE:
            pil_image.thumbnail((MatchingConfig.MAX_IMAGE_SIZE, MatchingConfig.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        # Enhance contrast for better feature extraction
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(MatchingConfig.CONTRAST_ENHANCEMENT_FACTOR)
        
        return pil_image
    
    def _extract_multiscale_features(self, pil_image: Image.Image) -> np.ndarray:
        """Extract features at multiple scales and combine them"""
        import torch
        
        all_features = []
        
        for scale in self.scales:
            try:
                # Resize image to current scale
                resized = pil_image.resize((scale, scale), Image.Resampling.LANCZOS)
                
                # Extract features at this scale
                scale_features = self._extract_features_at_scale(resized)
                all_features.append(scale_features)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract features at scale {scale}: {e}")
                # Add zero features as fallback with consistent dimensions
                fallback_size = self._get_single_scale_dimension()
                all_features.append(np.zeros(fallback_size, dtype=np.float32))
        
        # Combine features from different scales
        if all_features:
            combined_features = np.concatenate(all_features, axis=0)
            # L2 normalize the combined features
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            return combined_features
        else:
            # Fallback to single scale if all scales failed
            return self._extract_single_scale_features(pil_image)
    
    def _extract_single_scale_features(self, pil_image: Image.Image) -> np.ndarray:
        """Extract features at a single scale (original method with enhancements)"""
        # Use the standard input size for the model
        standard_size = 224
        resized = pil_image.resize((standard_size, standard_size), Image.Resampling.LANCZOS)
        return self._extract_features_at_scale(resized)
    
    def _extract_features_at_scale(self, pil_image: Image.Image) -> np.ndarray:
        """Extract ViT features at a specific scale with multi-layer support"""
        import torch
        
        # Preprocess the image
        inputs = self._extractor(images=pil_image, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            
            if self.use_multi_layer and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Multi-layer feature extraction
                features = self._extract_multi_layer_features(outputs)
            else:
                # Single-layer feature extraction (enhanced)
                features = self._extract_enhanced_single_layer_features(outputs)
            
            # L2 normalize
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            features = features.flatten()
            
            return features
    
    def _extract_multi_layer_features(self, outputs) -> np.ndarray:
        """Extract and combine features from multiple layers"""
        import torch
        
        # Get the last 4 layers for richer representation
        layer_indices = [-1, -2, -3, -4]
        layer_features = []
        
        for layer_idx in layer_indices:
            if abs(layer_idx) <= len(outputs.hidden_states):
                # Get CLS token from this layer
                layer_cls = outputs.hidden_states[layer_idx][:, 0]  # CLS token
                layer_features.append(layer_cls)
        
        if layer_features:
            # Concatenate features from different layers
            multi_layer_features = torch.cat(layer_features, dim=-1)
        else:
            # Fallback to last layer only
            multi_layer_features = outputs.last_hidden_state[:, 0]
        
        # Also include patch token features (averaged)
        patch_features = outputs.last_hidden_state[:, 1:].mean(dim=1)
        
        # Combine CLS tokens from multiple layers with patch features
        combined_features = torch.cat([multi_layer_features, patch_features], dim=-1)
        
        return combined_features.cpu().numpy()
    
    def _extract_enhanced_single_layer_features(self, outputs) -> np.ndarray:
        """Extract enhanced features from single layer (CLS + patch tokens)"""
        import torch
        
        # Get CLS token from last layer
        cls_token = outputs.last_hidden_state[:, 0]
        
        # Get patch tokens and apply different aggregation methods
        patch_tokens = outputs.last_hidden_state[:, 1:]
        
        # Multiple aggregation strategies for patch tokens
        patch_mean = patch_tokens.mean(dim=1)  # Global average pooling
        patch_max = patch_tokens.max(dim=1)[0]  # Global max pooling
        patch_std = patch_tokens.std(dim=1)    # Standard deviation pooling
        
        # Combine CLS token with different patch aggregations
        enhanced_features = torch.cat([
            cls_token,
            patch_mean,
            patch_max,
            patch_std
        ], dim=-1)
        
        return enhanced_features.cpu().numpy()
    
    def _get_single_scale_dimension(self) -> int:
        """Get feature dimension for single scale."""
        # Determine base dimension based on model
        if 'large' in self.model_name.lower():
            base_dim = MatchingConfig.VIT_LARGE_DIM
        else:
            base_dim = MatchingConfig.VIT_BASE_DIM
        
        if self.use_multi_layer:
            # Multi-layer: 4 layers of CLS tokens + 1 patch aggregation = 5 * base_dim
            return base_dim * 5
        else:
            # Enhanced single layer: CLS + mean + max + std of patches = 4 * base_dim
            return base_dim * 4
    
    def get_feature_dimension(self) -> int:
        """Calculate feature dimension based on configuration"""
        single_scale_dim = self._get_single_scale_dimension()
        
        if self.use_multi_scale:
            # Multi-scale: multiply by number of scales
            return single_scale_dim * len(self.scales)
        else:
            return single_scale_dim

class SimpleTileMatchingService:
    """Simplified matching service with basic image comparison algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate configuration weights
        self._validate_configuration()
        
        # Initialize advanced image preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Initialize Enhanced ViT feature extractors
        self.enhanced_vit_extractors = {}
        if VIT_AVAILABLE:
            try:
                # Standard Enhanced ViT (single-layer, single-scale)
                self.enhanced_vit_extractors['enhanced_single'] = EnhancedViTFeatureExtractor(
                    use_multi_layer=False, 
                    use_multi_scale=False
                )
                
                # Multi-layer Enhanced ViT
                self.enhanced_vit_extractors['enhanced_multi_layer'] = EnhancedViTFeatureExtractor(
                    use_multi_layer=True, 
                    use_multi_scale=False
                )
                
                # Multi-scale Enhanced ViT
                self.enhanced_vit_extractors['enhanced_multi_scale'] = EnhancedViTFeatureExtractor(
                    use_multi_layer=True, 
                    use_multi_scale=True
                )
                
                self.logger.info("Enhanced ViT feature extractors initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Enhanced ViT extractors: {e}")
                self.enhanced_vit_extractors = {}
        
        # Initialize ViT model if available (legacy support)
        self.vit_model = None
        self.vit_feature_extractor = None
        if VIT_AVAILABLE:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.logger.info(f"Initializing ViT model on device: {self.device}")
                
                model_name = "google/vit-base-patch16-224"
                self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
                self.vit_model = ViTModel.from_pretrained(model_name).to(self.device)
                self.vit_model.eval()
                
                self.logger.info("ViT model initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize ViT model: {e}")
                self.vit_model = None
                self.vit_feature_extractor = None
        else:
            self.logger.warning("ViT dependencies not available, using fallback methods")
    
    def _validate_configuration(self):
        """Validate configuration weights and parameters."""
        # Validate that all weight configurations sum to 1.0
        weight_configs = [
            ('ENSEMBLE_WEIGHTS', MatchingConfig.ENSEMBLE_WEIGHTS),
            ('ENHANCED_ENSEMBLE_WEIGHTS', MatchingConfig.ENHANCED_ENSEMBLE_WEIGHTS),
            ('ADVANCED_PREPROCESSING_WEIGHTS', MatchingConfig.ADVANCED_PREPROCESSING_WEIGHTS),
            ('VIT_SIMILARITY_WEIGHTS', MatchingConfig.VIT_SIMILARITY_WEIGHTS)
        ]
        
        for name, weights in weight_configs:
            if not MatchingConfig.validate_weights(weights):
                self.logger.warning(f"Configuration {name} weights do not sum to 1.0: {sum(weights.values())}")
        
        self.logger.info("Configuration validation completed")
    
    def _preprocess_image(self, image_data: bytes, enable_advanced_preprocessing: bool = False, 
                         enable_tile_extraction: bool = False) -> np.ndarray:
        """Convert image bytes to OpenCV format and apply advanced preprocessing."""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(BytesIO(image_data))
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array
            img_array = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Apply advanced preprocessing if enabled
            if enable_advanced_preprocessing:
                try:
                    img_bgr = self.preprocessor.preprocess(img_bgr, enable_tile_extraction=enable_tile_extraction)
                    self.logger.debug("Advanced preprocessing applied successfully")
                except Exception as e:
                    self.logger.warning(f"Advanced preprocessing failed, using basic preprocessing: {e}")
                    # Continue with basic preprocessing if advanced fails
            
            return img_bgr
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def _calculate_color_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using color histogram comparison."""
        try:
            # Calculate histograms for each channel
            hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
            hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])
            
            hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
            hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
            hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])
            
            # Compare histograms using correlation
            corr_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
            corr_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
            corr_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
            
            # Average the correlations
            similarity = (corr_b + corr_g + corr_r) / 3.0
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error calculating color histogram similarity: {e}")
            return 0.0
    
    def _calculate_orb_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using ORB feature matching."""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=500)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return 0.0
            
            # Create BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match descriptors
            matches = bf.match(des1, des2)
            
            if len(matches) == 0:
                return 0.0
            
            # Sort matches by distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate similarity based on good matches
            good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches
            max_features = max(len(kp1), len(kp2))
            
            if max_features == 0:
                return 0.0
            
            similarity = len(good_matches) / max_features
            return min(1.0, similarity)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ORB similarity: {e}")
            return 0.0
    
    def _extract_vit_features_proven(self, image_data: bytes) -> np.ndarray:
        """Extract ViT features using the proven Hugging Face method."""
        try:
            if self.vit_model is None or self.vit_feature_extractor is None:
                self.logger.warning("ViT model not available, using fallback")
                return np.zeros(768)  # Standard ViT feature dimension
            
            # Convert bytes to PIL Image
            pil_image = Image.open(BytesIO(image_data)).convert("RGB")
            
            # Use the proven feature extraction method
            inputs = self.vit_feature_extractor(images=pil_image, return_tensors="pt").to(self.device)
            
            # Extract image features using the proven method
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                # Use the CLS token as the image representation (proven approach)
                features = outputs.last_hidden_state[:, 0].cpu().numpy()
                # Normalize features (proven approach)
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            return features.flatten()
            
        except Exception as e:
            self.logger.error(f"Error extracting proven ViT features: {e}")
            return np.zeros(768)
    
    def _extract_patch_embeddings(self, img: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """Extract patch embeddings similar to ViT preprocessing."""
        try:
            # Resize to standard ViT input size
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Extract non-overlapping patches
            patches = []
            num_patches = 224 // patch_size
            
            for i in range(num_patches):
                for j in range(num_patches):
                    y_start, y_end = i * patch_size, (i + 1) * patch_size
                    x_start, x_end = j * patch_size, (j + 1) * patch_size
                    patch = img_rgb[y_start:y_end, x_start:x_end]
                    
                    # Flatten patch and add positional encoding
                    patch_flat = patch.flatten()
                    
                    # Simple positional encoding
                    pos_embed = np.array([
                        np.sin(i / 10.0), np.cos(i / 10.0),
                        np.sin(j / 10.0), np.cos(j / 10.0)
                    ])
                    
                    # Combine patch with position
                    patch_with_pos = np.concatenate([patch_flat, pos_embed])
                    patches.append(patch_with_pos)
            
            return np.array(patches)
            
        except Exception as e:
            self.logger.error(f"Error extracting patch embeddings: {e}")
            return np.zeros((196, 768))  # 14x14 patches, 768 dim
    
    def _compute_attention_weights(self, patch_embeddings: np.ndarray) -> np.ndarray:
        """Compute simplified attention weights between patches."""
        try:
            num_patches = patch_embeddings.shape[0]
            
            # Simple attention: compute similarity between all patch pairs
            attention_matrix = np.zeros((num_patches, num_patches))
            
            for i in range(num_patches):
                for j in range(num_patches):
                    # Cosine similarity between patches
                    dot_product = np.dot(patch_embeddings[i], patch_embeddings[j])
                    norm_i = np.linalg.norm(patch_embeddings[i])
                    norm_j = np.linalg.norm(patch_embeddings[j])
                    
                    if norm_i > 0 and norm_j > 0:
                        attention_matrix[i, j] = dot_product / (norm_i * norm_j)
            
            # Softmax-like normalization
            attention_matrix = np.exp(attention_matrix)
            attention_matrix = attention_matrix / (np.sum(attention_matrix, axis=1, keepdims=True) + 1e-8)
            
            return attention_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing attention weights: {e}")
            return np.eye(patch_embeddings.shape[0])
    
    def _extract_vit_features(self, img: np.ndarray) -> np.ndarray:
        """Extract advanced ViT-like features with attention mechanisms."""
        try:
            # Extract patch embeddings
            patch_embeddings = self._extract_patch_embeddings(img)
            
            # Compute attention weights
            attention_weights = self._compute_attention_weights(patch_embeddings)
            
            # Apply attention to get attended patch features
            attended_patches = np.matmul(attention_weights, patch_embeddings)
            
            # Global features from attended patches
            global_feature = np.mean(attended_patches, axis=0)
            attention_variance = np.var(attended_patches, axis=0)
            
            # Key patch features (patches with highest attention)
            attention_scores = np.sum(attention_weights, axis=1)
            top_patch_indices = np.argsort(attention_scores)[-10:]  # Top 10 patches
            key_patches = attended_patches[top_patch_indices]
            key_features = np.mean(key_patches, axis=0)
            
            # Spatial relationship features
            spatial_features = []
            num_patches_per_side = int(np.sqrt(patch_embeddings.shape[0]))
            
            for i in range(num_patches_per_side):
                for j in range(num_patches_per_side):
                    patch_idx = i * num_patches_per_side + j
                    
                    # Attention to neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < num_patches_per_side and 0 <= nj < num_patches_per_side:
                                neighbor_idx = ni * num_patches_per_side + nj
                                neighbors.append(attention_weights[patch_idx, neighbor_idx])
                    
                    if neighbors:
                        spatial_features.append(np.mean(neighbors))
            
            # Hierarchical features (multi-scale attention)
            hierarchical_features = []
            
            # Group patches into 4x4 regions and compute region-level features
            region_size = num_patches_per_side // 4
            for ri in range(4):
                for rj in range(4):
                    region_patches = []
                    for i in range(ri * region_size, (ri + 1) * region_size):
                        for j in range(rj * region_size, (rj + 1) * region_size):
                            if i < num_patches_per_side and j < num_patches_per_side:
                                patch_idx = i * num_patches_per_side + j
                                region_patches.append(attended_patches[patch_idx])
                    
                    if region_patches:
                        region_feature = np.mean(region_patches, axis=0)
                        hierarchical_features.append(region_feature[:50])  # Limit size
            
            # Combine all features
            feature_vector = np.concatenate([
                global_feature[:200],  # Global attended features
                attention_variance[:100],  # Attention variance
                key_features[:150],  # Key patch features
                spatial_features[:100],  # Spatial relationship features
                np.array(hierarchical_features).flatten()[:200]  # Hierarchical features
            ])
            
            # Normalize features
            feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error extracting ViT features: {e}")
            return np.zeros(750)  # Return zero vector on error
    
    def _calculate_vit_similarity_proven(self, img1_data: bytes, img2_data: bytes) -> float:
        """Calculate similarity using the proven ViT method from external matcher."""
        try:
            # Extract features using the proven method
            features1 = self._extract_vit_features_proven(img1_data)
            features2 = self._extract_vit_features_proven(img2_data)
            
            # Use the proven similarity calculation - cosine similarity
            # This is exactly what the proven vit_matcher.py uses
            similarity = np.dot(features1, features2)
            
            # Ensure similarity is in valid range [0, 1]
            similarity = max(0.0, min(1.0, float(similarity)))
            
            self.logger.debug(f"Proven ViT similarity: {similarity}")
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating proven ViT similarity: {e}")
            return 0.0
    
    def _calculate_enhanced_vit_similarity(self, img1_data: bytes, img2_data: bytes, extractor_type: str = 'enhanced_multi_layer') -> float:
        """Calculate similarity using Enhanced ViT feature extractor."""
        try:
            if extractor_type not in self.enhanced_vit_extractors:
                self.logger.warning(f"Enhanced ViT extractor '{extractor_type}' not available, falling back to proven method")
                return self._calculate_vit_similarity_proven(img1_data, img2_data)
            
            extractor = self.enhanced_vit_extractors[extractor_type]
            
            # Extract enhanced features
            features1 = extractor.extract_features(img1_data)
            features2 = extractor.extract_features(img2_data)
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8)
            
            # Ensure similarity is in valid range [0, 1]
            similarity = max(0.0, min(1.0, float(similarity)))
            
            self.logger.debug(f"Enhanced ViT ({extractor_type}) similarity: {similarity}")
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating Enhanced ViT similarity: {e}")
            # Fallback to proven method
            return self._calculate_vit_similarity_proven(img1_data, img2_data)
    
    def _calculate_vit_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using ViT-like feature extraction with enhanced discrimination."""
        try:
            # Extract features from both images
            features1 = self._extract_vit_features(img1)
            features2 = self._extract_vit_features(img2)
            
            # Calculate multiple similarity measures
            
            # 1. Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            
            # 2. Euclidean distance (inverted and normalized)
            euclidean_distance = np.linalg.norm(features1 - features2)
            max_distance = np.linalg.norm(features1) + np.linalg.norm(features2)
            euclidean_similarity = 1.0 - (euclidean_distance / (max_distance + 1e-8))
            
            # 3. Pearson correlation coefficient
            try:
                correlation = np.corrcoef(features1, features2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            
            # 4. Manhattan distance (inverted and normalized)
            manhattan_distance = np.sum(np.abs(features1 - features2))
            max_manhattan = np.sum(np.abs(features1)) + np.sum(np.abs(features2))
            manhattan_similarity = 1.0 - (manhattan_distance / (max_manhattan + 1e-8))
            
            # Combine similarities with configured weights
            weights = MatchingConfig.VIT_SIMILARITY_WEIGHTS
            combined_similarity = (
                weights['cosine'] * cosine_similarity +
                weights['euclidean'] * euclidean_similarity +
                weights['correlation'] * correlation +
                weights['manhattan'] * manhattan_similarity
            )
            
            # Apply sigmoid-like transformation to make it more discriminative
            # This makes very similar images score higher and dissimilar images score lower
            discriminative_similarity = 1.0 / (1.0 + np.exp(-10 * (combined_similarity - 0.5)))
            
            # Convert to 0-1 range and apply final adjustments
            final_similarity = (discriminative_similarity + 1) / 2
            
            # Apply threshold-based adjustment for better discrimination
            if final_similarity > 0.7:
                # Boost high similarities
                final_similarity = 0.7 + 0.3 * ((final_similarity - 0.7) / 0.3) ** 0.5
            elif final_similarity < 0.3:
                # Reduce low similarities
                final_similarity = final_similarity ** 2
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            self.logger.error(f"Error calculating ViT similarity: {e}")
            return 0.0
    
    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using structural similarity (simplified SSIM)."""
        try:
            # Resize images to same size for comparison
            height, width = 128, 128  # Standard size for comparison
            img1_resized = cv2.resize(img1, (width, height))
            img2_resized = cv2.resize(img2, (width, height))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean squared error (lower is more similar)
            mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            
            # Convert MSE to similarity score (0-1, where 1 is identical)
            max_mse = 255 ** 2  # Maximum possible MSE
            similarity = 1.0 - (mse / max_mse)
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating structural similarity: {e}")
            return 0.0
    
    async def find_similar_tiles(self, query_image_data: bytes, top_k: int = 5, 
                               method: str = 'color_hist', threshold: float = 0.0,
                               enable_advanced_preprocessing: bool = False,
                               enable_tile_extraction: bool = False) -> List[Dict[str, Any]]:
        """
        Find similar tiles using actual image comparison algorithms.
        """
        self.logger.info(f"Image matching called with top_k={top_k}, method={method}, threshold={threshold}")
        self.logger.info(f"Advanced preprocessing: {enable_advanced_preprocessing}, Tile extraction: {enable_tile_extraction}")
        try:
            # Preprocess query image
            query_img = self._preprocess_image(query_image_data, enable_advanced_preprocessing, enable_tile_extraction)
            
            # Optimize database query based on requirements
            db_limit = max(top_k * 2, MatchingConfig.DEFAULT_DB_LIMIT)  # Get at least 2x requested results
            tiles = await Tile.find().limit(db_limit).to_list()
            self.logger.info(f"Found {len(tiles)} tiles in database for comparison (limit: {db_limit})")
            
            if not tiles:
                self.logger.warning("No tiles found in database")
                return []
            
            # Calculate similarities
            similarities = []
            
            for tile in tiles:
                if not tile.image_data:
                    continue
                    
                try:
                    # Calculate similarity based on method
                    if method == 'color_hist':
                        # Preprocess tile image for traditional methods
                        tile_img = self._preprocess_image(tile.image_data, enable_advanced_preprocessing, enable_tile_extraction)
                        similarity = self._calculate_color_histogram_similarity(query_img, tile_img)
                    elif method == 'orb':
                        # Preprocess tile image for traditional methods
                        tile_img = self._preprocess_image(tile.image_data, enable_advanced_preprocessing, enable_tile_extraction)
                        similarity = self._calculate_orb_similarity(query_img, tile_img)
                    elif method in ['vit_simple', 'vit_multi_layer', 'vit_multi_scale']:
                        # For ViT methods, use raw image bytes for proven method
                        # but preprocess for traditional ViT feature extraction
                        similarity = self._calculate_vit_similarity_proven(query_image_data, tile.image_data)
                    elif method == 'enhanced_vit_single':
                        # Enhanced ViT with single layer, single scale
                        similarity = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_single')
                    elif method == 'enhanced_vit_multi_layer':
                        # Enhanced ViT with multi-layer features
                        similarity = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_multi_layer')
                    elif method == 'enhanced_vit_multi_scale':
                        # Enhanced ViT with multi-layer and multi-scale features
                        similarity = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_multi_scale')
                    elif method == 'ensemble':
                        # Combine multiple methods with Enhanced ViT using configured weights
                        tile_img = self._preprocess_image(tile.image_data, enable_advanced_preprocessing, enable_tile_extraction)
                        hist_sim = self._calculate_color_histogram_similarity(query_img, tile_img)
                        orb_sim = self._calculate_orb_similarity(query_img, tile_img)
                        enhanced_vit_sim = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_multi_layer')
                        
                        weights = MatchingConfig.ENSEMBLE_WEIGHTS
                        similarity = (hist_sim * weights['hist'] + 
                                    orb_sim * weights['orb'] + 
                                    enhanced_vit_sim * weights['enhanced_vit'])
                    elif method == 'enhanced_ensemble':
                        # Advanced ensemble with multiple Enhanced ViT extractors
                        tile_img = self._preprocess_image(tile.image_data, enable_advanced_preprocessing, enable_tile_extraction)
                        hist_sim = self._calculate_color_histogram_similarity(query_img, tile_img)
                        orb_sim = self._calculate_orb_similarity(query_img, tile_img)
                        enhanced_single_sim = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_single')
                        enhanced_multi_layer_sim = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_multi_layer')
                        enhanced_multi_scale_sim = self._calculate_enhanced_vit_similarity(query_image_data, tile.image_data, 'enhanced_multi_scale')
                        
                        weights = MatchingConfig.ENHANCED_ENSEMBLE_WEIGHTS
                        similarity = (hist_sim * weights['hist'] + 
                                    orb_sim * weights['orb'] + 
                                    enhanced_single_sim * weights['enhanced_single'] + 
                                    enhanced_multi_layer_sim * weights['enhanced_multi_layer'] + 
                                    enhanced_multi_scale_sim * weights['enhanced_multi_scale'])
                    elif method == 'advanced_preprocessing':
                        # Use advanced preprocessing with tile extraction and ensemble matching
                        tile_img = self._preprocess_image(tile.image_data, True, True)  # Force advanced preprocessing and tile extraction
                        query_img_extracted = self._preprocess_image(query_image_data, True, True)  # Apply same to query
                        hist_sim = self._calculate_color_histogram_similarity(query_img_extracted, tile_img)
                        orb_sim = self._calculate_orb_similarity(query_img_extracted, tile_img)
                        vit_sim = self._calculate_vit_similarity_proven(query_image_data, tile.image_data)
                        
                        weights = MatchingConfig.ADVANCED_PREPROCESSING_WEIGHTS
                        similarity = (hist_sim * weights['hist'] + 
                                    orb_sim * weights['orb'] + 
                                    vit_sim * weights['vit'])
                    else:
                        # Default to color histogram
                        tile_img = self._preprocess_image(tile.image_data, enable_advanced_preprocessing, enable_tile_extraction)
                        similarity = self._calculate_color_histogram_similarity(query_img, tile_img)
                    
                    # Only include if above threshold
                    if similarity >= threshold:
                        similarities.append((tile, similarity))
                        
                except Exception as e:
                    self.logger.warning(f"Error processing tile {tile.sku}: {e}")
                    continue
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k results
            top_similarities = similarities[:top_k]
            
            # Format results
            results = []
            for tile, similarity in top_similarities:
                results.append({
                    'tile_id': str(tile.id),
                    'similarity': similarity,
                    'metadata': {
                        'sku': tile.sku,
                        'model_name': tile.model_name,
                        'collection_name': tile.collection_name,
                        'content_type': tile.content_type,
                        'image_path': tile.image_path,
                        'uploaded_at': str(tile.created_at) if tile.created_at else None
                    },
                    'created_at': tile.created_at.timestamp() if tile.created_at else None,
                    'has_image_data': bool(tile.image_data),
                    'image_data': base64.b64encode(tile.image_data).decode('utf-8') if tile.image_data else None
                })
            
            self.logger.info(f"Returning {len(results)} actual matches using {method}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in image matching: {e}")
            # Don't swallow all exceptions - let database errors propagate
            raise
    
    def get_available_methods(self) -> List[str]:
        """Return list of available methods."""
        base_methods = ['color_hist', 'orb', 'vit_simple', 'vit_multi_layer', 'vit_multi_scale', 'ensemble', 'advanced_preprocessing']
        
        # Add Enhanced ViT methods if available
        if self.enhanced_vit_extractors:
            enhanced_methods = ['enhanced_vit_single', 'enhanced_vit_multi_layer', 'enhanced_vit_multi_scale', 'enhanced_ensemble']
            base_methods.extend(enhanced_methods)
        
        return base_methods
    
    def get_tile_image_data(self, tile_id: str) -> Optional[Dict[str, Any]]:
        """Get tile image data by ID - synchronous version for compatibility."""
        try:
            # This is a sync method for compatibility with existing code
            # In a real async implementation, you'd use the async version
            return None  # Will be handled by the async route logic
        except Exception as e:
            self.logger.error(f"Error getting tile image data: {e}")
            return None
    
    async def get_tile_image_data_async(self, tile_id: str) -> Optional[Dict[str, Any]]:
        """Get tile image data by ID - async version."""
        try:
            from bson import ObjectId
            
            # Try to get tile from database
            tile = await Tile.get(ObjectId(tile_id))
            if tile and tile.image_data:
                return {
                    'tile_id': str(tile.id),
                    'content_type': tile.content_type or 'image/jpeg',
                    'data': base64.b64encode(tile.image_data).decode('utf-8')
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting tile image data: {e}")
            return None
    
    def add_tile(self, image_data: bytes, metadata: Dict[str, Any], tile_id: str) -> None:
        """
        Add a tile to the matching service.
        
        This method is called when a tile is uploaded to the catalog.
        Since this service is stateless and uses the database directly,
        we don't need to maintain an in-memory index.
        
        Args:
            image_data: The tile image data as bytes
            metadata: Tile metadata dictionary
            tile_id: The tile ID
        """
        try:
            self.logger.info(f"Tile {tile_id} added to matching service with metadata: {metadata.get('sku', 'Unknown')}")
            # In this implementation, we don't need to do anything special
            # as the matching service queries the database directly
            
        except Exception as e:
            self.logger.error(f"Error adding tile to matching service: {e}")
            # Don't raise the exception as this is not critical for the upload process

# Global service instance
_simple_matching_service = None

def get_simple_matching_service() -> SimpleTileMatchingService:
    """Get or create the simple matching service instance."""
    global _simple_matching_service
    if _simple_matching_service is None:
        _simple_matching_service = SimpleTileMatchingService()
    return _simple_matching_service