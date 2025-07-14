from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO
import os
import io
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import hashlib
import base64
import json
import asyncio
import concurrent.futures
from collections import defaultdict
from abc import ABC, abstractmethod
import threading
import warnings
from dataclasses import dataclass
from enum import Enum
from bson import ObjectId

# Database models
from backend.models.tile import Tile, FeatureVector

# Optional advanced dependencies
try:
    import torch
    import torch.nn.functional as F
    from transformers import ViTImageProcessor, ViTModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/Transformers not available. VIT features will be disabled.")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Sklearn not available. Some advanced features will be disabled.")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP module not available. CLIP feature extraction will be disabled.")

# Configuration constants
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB max image size
MAX_CACHE_SIZE = 1000  # Maximum number of tiles to cache in memory
VOCAB_SIZE = 100  # Size of vocabulary for bag-of-words
DEFAULT_FEATURE_DIM = 512

# Feature dimensions for different extractors
FEATURE_DIMENSIONS = {
    'color_hist': 512,
    'orb': 512,
    'vit_simple': 768,
    'vit_multi_layer': 768,
    'vit_multi_scale': 768,
    'sift': 512,
    'kaze': 512,
    'clip': 512,
    'ensemble': 1024  # Combined features
}

class SimilarityMetric(Enum):
    """Enumeration of available similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    PEARSON = "pearson"
    BHATTACHARYYA = "bhattacharyya"
    JENSEN_SHANNON = "jensen_shannon"

@dataclass
class MatchResult:
    """Result of a similarity match."""
    tile_id: str
    similarity: float
    confidence: float
    method: str
    metadata: Dict[str, Any]

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM):
        self.feature_dim = feature_dim
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the feature extractor."""
        pass

class ColorHistogramExtractor(FeatureExtractor):
    """Extract color histogram features."""
    
    def __init__(self, feature_dim: int = 512, bins: int = 8):
        super().__init__(feature_dim)
        self.bins = bins
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            
            # Calculate histogram for each channel
            hist_r = cv2.calcHist([img_rgb], [0], None, [self.bins], [0, 256])
            hist_g = cv2.calcHist([img_rgb], [1], None, [self.bins], [0, 256])
            hist_b = cv2.calcHist([img_rgb], [2], None, [self.bins], [0, 256])
            
            # Concatenate and normalize
            features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            features = features / (np.sum(features) + 1e-8)  # Normalize
            
            # Pad or truncate to target dimension
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
                
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting color histogram: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def get_name(self) -> str:
        return "color_hist"

class ORBExtractor(FeatureExtractor):
    """Extract ORB features using bag-of-words approach."""
    
    def __init__(self, feature_dim: int = 512, n_features: int = 500):
        super().__init__(feature_dim)
        self.n_features = n_features
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.vocabulary = None
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract ORB features."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # Use statistical features if vocabulary not available
            features = self._compute_statistical_features(descriptors)
            
            # Pad or truncate to target dimension
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
                
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting ORB features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _compute_statistical_features(self, descriptors: np.ndarray) -> np.ndarray:
        """Compute statistical features from descriptors."""
        features = []
        features.extend(np.mean(descriptors, axis=0))
        features.extend(np.std(descriptors, axis=0))
        features.extend(np.max(descriptors, axis=0))
        features.extend(np.min(descriptors, axis=0))
        return np.array(features)
    
    def get_name(self) -> str:
        return "orb"

class VisionTransformerExtractor(FeatureExtractor):
    """Vision Transformer feature extractor with multiple variants."""
    
    def __init__(self, feature_dim: int = 768, variant: str = 'simple', model_name: str = 'google/vit-base-patch16-224'):
        super().__init__(feature_dim)
        self.variant = variant
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if TORCH_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ViT model and processor."""
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Initialized ViT model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ViT model: {e}")
            self.model = None
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract ViT features based on variant."""
        if not TORCH_AVAILABLE or self.model is None:
            return self._extract_simple_patch_features(image)
        
        try:
            if self.variant == 'multi_layer':
                return self._extract_multi_layer_features(image)
            elif self.variant == 'multi_scale':
                return self._extract_multi_scale_features(image)
            else:
                return self._extract_simple_vit_features(image)
                
        except Exception as e:
            self.logger.error(f"Error extracting ViT features: {e}")
            return self._extract_simple_patch_features(image)
    
    def _extract_simple_vit_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using standard ViT."""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0]  # CLS token
                features = features.cpu().numpy().flatten()
            
            # Adjust to target dimension
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)))
                else:
                    features = features[:self.feature_dim]
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in simple ViT extraction: {e}")
            return self._extract_simple_patch_features(image)
    
    def _extract_multi_layer_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from multiple layers and combine."""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Combine features from last 4 layers
                hidden_states = outputs.hidden_states[-4:]
                combined_features = []
                
                for hidden_state in hidden_states:
                    cls_token = hidden_state[:, 0].cpu().numpy().flatten()
                    combined_features.append(cls_token)
                
                # Average the features
                features = np.mean(combined_features, axis=0)
            
            # Adjust to target dimension
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)))
                else:
                    features = features[:self.feature_dim]
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in multi-layer ViT extraction: {e}")
            return self._extract_simple_patch_features(image)
    
    def _extract_multi_scale_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features at multiple scales and combine."""
        try:
            scales = [224, 256, 288]  # Different input sizes
            all_features = []
            
            for scale in scales:
                # Resize image
                resized = cv2.resize(image, (scale, scale))
                
                if len(resized.shape) == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(resized)
                
                inputs = self.processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
                    all_features.append(features)
            
            # Average features across scales
            features = np.mean(all_features, axis=0)
            
            # Adjust to target dimension
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)))
                else:
                    features = features[:self.feature_dim]
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in multi-scale ViT extraction: {e}")
            return self._extract_simple_patch_features(image)
    
    def _extract_simple_patch_features(self, image: np.ndarray) -> np.ndarray:
        """Fallback patch-based feature extraction."""
        try:
            # Resize to standard size
            img_resized = cv2.resize(image, (224, 224))
            
            # Extract patch features
            patch_size = 16
            patches = []
            
            for i in range(0, 224, patch_size):
                for j in range(0, 224, patch_size):
                    patch = img_resized[i:i+patch_size, j:j+patch_size]
                    patch_features = [
                        np.mean(patch),
                        np.std(patch),
                        np.max(patch),
                        np.min(patch)
                    ]
                    patches.extend(patch_features)
            
            features = np.array(patches)
            
            # Adjust to target dimension
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in patch feature extraction: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def get_name(self) -> str:
        return f"vit_{self.variant}"

class EnsembleExtractor(FeatureExtractor):
    """Ensemble feature extractor combining multiple extractors."""
    
    def __init__(self, extractors: List[FeatureExtractor], feature_dim: int = 1024):
        super().__init__(feature_dim)
        self.extractors = extractors
        self.weights = np.ones(len(extractors)) / len(extractors)  # Equal weights initially
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features using all extractors and combine."""
        try:
            all_features = []
            
            for extractor in self.extractors:
                features = extractor.extract(image)
                all_features.append(features)
            
            # Weighted combination
            combined = np.concatenate(all_features)
            
            # Adjust to target dimension
            if len(combined) < self.feature_dim:
                combined = np.pad(combined, (0, self.feature_dim - len(combined)))
            else:
                combined = combined[:self.feature_dim]
            
            return combined.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in ensemble extraction: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def get_name(self) -> str:
        extractor_names = [ext.get_name() for ext in self.extractors]
        return f"ensemble_{'_'.join(extractor_names)}"

class SimilarityCalculator:
    """Advanced similarity calculator with multiple metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                           metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """Calculate similarity between two feature vectors."""
        try:
            if metric == SimilarityMetric.COSINE:
                return self._cosine_similarity(features1, features2)
            elif metric == SimilarityMetric.EUCLIDEAN:
                return self._euclidean_similarity(features1, features2)
            elif metric == SimilarityMetric.MANHATTAN:
                return self._manhattan_similarity(features1, features2)
            elif metric == SimilarityMetric.PEARSON:
                return self._pearson_similarity(features1, features2)
            elif metric == SimilarityMetric.BHATTACHARYYA:
                return self._bhattacharyya_similarity(features1, features2)
            elif metric == SimilarityMetric.JENSEN_SHANNON:
                return self._jensen_shannon_similarity(features1, features2)
            else:
                return self._cosine_similarity(features1, features2)
                
        except Exception as e:
            self.logger.error(f"Error calculating {metric.value} similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def _euclidean_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate normalized euclidean similarity."""
        distance = np.linalg.norm(f1 - f2)
        # Convert distance to similarity (higher is better)
        max_distance = np.sqrt(len(f1))  # Maximum possible L2 distance for normalized vectors
        return float(1.0 - min(distance / max_distance, 1.0))
    
    def _manhattan_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate normalized manhattan similarity."""
        distance = np.sum(np.abs(f1 - f2))
        max_distance = len(f1)  # Maximum possible L1 distance for normalized vectors
        return float(1.0 - min(distance / max_distance, 1.0))
    
    def _pearson_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        if np.std(f1) == 0 or np.std(f2) == 0:
            return 0.0
        correlation = np.corrcoef(f1, f2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _bhattacharyya_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate Bhattacharyya similarity."""
        # Normalize to probability distributions
        f1_norm = f1 / (np.sum(f1) + 1e-8)
        f2_norm = f2 / (np.sum(f2) + 1e-8)
        
        # Calculate Bhattacharyya coefficient
        bc = np.sum(np.sqrt(f1_norm * f2_norm))
        return float(bc)
    
    def _jensen_shannon_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate Jensen-Shannon similarity."""
        # Normalize to probability distributions
        f1_norm = f1 / (np.sum(f1) + 1e-8)
        f2_norm = f2 / (np.sum(f2) + 1e-8)
        
        # Calculate Jensen-Shannon divergence
        m = 0.5 * (f1_norm + f2_norm)
        
        def kl_divergence(p, q):
            return np.sum(p * np.log((p + 1e-8) / (q + 1e-8)))
        
        js_divergence = 0.5 * kl_divergence(f1_norm, m) + 0.5 * kl_divergence(f2_norm, m)
        js_distance = np.sqrt(js_divergence)
        
        # Convert to similarity
        return float(1.0 - min(js_distance, 1.0))
    
    def calculate_ensemble_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                                   weights: Optional[List[float]] = None) -> Tuple[float, float]:
        """Calculate ensemble similarity using multiple metrics.
        
        Returns:
            Tuple of (similarity_score, confidence)
        """
        metrics = [SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN, 
                  SimilarityMetric.MANHATTAN, SimilarityMetric.PEARSON]
        
        similarities = []
        for metric in metrics:
            sim = self.calculate_similarity(features1, features2, metric)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Use provided weights or adaptive weights
        if weights is None:
            # Adaptive weighting based on consistency
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            
            # Higher weight for metrics closer to mean (more consistent)
            weights = np.exp(-0.5 * ((similarities - mean_sim) / (std_sim + 1e-8))**2)
            weights = weights / np.sum(weights)
        
        # Calculate weighted similarity
        weighted_similarity = np.sum(similarities * weights)
        
        # Calculate confidence based on agreement between metrics
        confidence = 1.0 - np.std(similarities)
        confidence = max(0.0, min(1.0, confidence))
        
        return float(weighted_similarity), float(confidence)

class TileMatchingService:
    """Enhanced tile matching service with database storage and advanced feature extractors."""
    
    def __init__(self, max_cache_size: int = MAX_CACHE_SIZE):
        """Initialize the enhanced tile matching service with database backend.
        
        Args:
            max_cache_size: Maximum number of tiles to cache in memory.
        """
        self.feature_cache: Dict[str, Dict[str, np.ndarray]] = {}  # tile_id -> {method: features}
        self.max_cache_size = max_cache_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize feature extractors and similarity calculator
        self.extractors: Dict[str, FeatureExtractor] = {}
        self.similarity_calculator = SimilarityCalculator()
        
        # Initialize tiles dictionary for in-memory storage
        self.tiles: Dict[str, Dict[str, Any]] = {}
        
        # Initialize feature extraction components
        self._initialize_extractors()
        
        self.logger.info("Initialized Enhanced TileMatchingService with database backend")
        self.logger.info(f"Available extractors: {list(self.extractors.keys())}")
        
        # The _preload_task will be created and assigned by the get_matching_service dependency.
        self._preload_task: Optional[asyncio.Task] = None
    
    def _initialize_extractors(self):
        """Initialize available feature extractors."""
        # Initialize basic extractors that are always available
        self.extractors['color_hist'] = 'ColorHistogramExtractor'
        self.extractors['orb'] = 'ORBExtractor'
        
        # Initialize ViT extractors if PyTorch is available
        if TORCH_AVAILABLE:
            self.extractors['vit_simple'] = 'SimpleViTExtractor'
            self.extractors['vit_multi_layer'] = 'MultiLayerViTExtractor'
            self.extractors['vit_multi_scale'] = 'MultiScaleViTExtractor'
        
        # Initialize CLIP if available
        if CLIP_AVAILABLE:
            self.extractors['clip'] = 'CLIPExtractor'
        
        # Ensemble is always available if we have multiple extractors
        if len(self.extractors) > 1:
            self.extractors['ensemble'] = 'EnsembleExtractor'
    
    async def _preload_feature_cache(self) -> None:
        """Preload feature cache from database for faster access.
        
        This method runs asynchronously in the background after service initialization.
        It loads feature vectors for the most recently used tiles up to the cache limit.
        """
        try:
            self.logger.info("Starting feature cache preloading...")
            start_time = time.time()
            
            # Check if Beanie is properly initialized
            if not hasattr(Tile, '_inheritance_inited'):
                self.logger.warning("Beanie not fully initialized yet, skipping cache preload")
                return
            
            # Get most recently used tiles up to cache limit
            tiles = await Tile.find().sort([("updated_at", -1)]).limit(self.max_cache_size).to_list()
            
            if not tiles:
                self.logger.info("No tiles found in the database to preload")
                return
                
            loaded_count = 0
            for tile in tiles:
                try:
                    tile_id = str(tile.id)
                    # Only cache if not already in cache
                    if tile_id not in self.feature_cache:
                        # Initialize cache entry for this tile
                        self.feature_cache[tile_id] = {}
                        
                        # Cache each feature vector
                        if hasattr(tile, 'features') and tile.features:
                            # Convert FeatureVector model to dictionary for iteration
                            features_dict = tile.features.dict(exclude_none=True)
                            for method, vector in features_dict.items():
                                if vector:  # Only cache non-empty vectors
                                    self.feature_cache[tile_id][method] = np.array(vector)
                            
                            loaded_count += 1
                            
                            # Log progress periodically
                            if loaded_count % 10 == 0:  # Log every 10 tiles to avoid too many logs
                                self.logger.debug(f"Preloaded features for {loaded_count}/{len(tiles)} tiles...")
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load features for tile {getattr(tile, 'id', 'unknown')}: {e}",
                        exc_info=self.logger.isEnabledFor(logging.DEBUG)
                    )
            
            # Log completion
            duration = time.time() - start_time
            self.logger.info(
                f"Preloaded features for {loaded_count}/{len(tiles)} tiles "
                f"in {duration:.2f} seconds"
            )
            
        except Exception as e:
            self.logger.error(f"Error preloading feature cache: {e}", exc_info=True)
    
    def _setup_upload_directory(self, upload_dir: str = None) -> Path:
        """Setup and validate upload directory."""
        try:
            if upload_dir is None:
                # Default to backend/uploads relative to this file
                current_dir = Path(__file__).parent
                upload_path = current_dir.parent / 'uploads'
            else:
                upload_path = Path(upload_dir).resolve()  # Fix: resolve to prevent path traversal
            
            # Create directory if it doesn't exist
            upload_path.mkdir(parents=True, exist_ok=True)
            
            # Verify directory is writable
            test_file = upload_path / '.test_write'
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise PermissionError(f"Upload directory not writable: {upload_path}")
            
            return upload_path
            
           
            
        except Exception as e:
            self.logger.error(f"Error setting up upload directory: {e}")
            raise
    
    def _generate_content_hash(self, image_data: bytes) -> str:
        """Generate content-based hash for image data."""
        if not image_data:
            raise ValueError("Empty image data")
        return hashlib.sha256(image_data).hexdigest()[:16]
    
    def _validate_image_data(self, image_data: bytes) -> bool:
        """Validate that image data is readable and within size limits."""
        try:
            # Check size limits
            if len(image_data) > MAX_IMAGE_SIZE:
                self.logger.warning(f"Image too large: {len(image_data)} bytes")
                return False
                
            if len(image_data) == 0:
                return False
                
            # Try to decode with PIL first
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            return True
        except Exception:
            try:
                # Try with OpenCV
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img is not None and img.size > 0
            except Exception:
                return False
    
    def _initialize_extractors(self):
        """Initialize feature extraction components."""
        # Always available extractors
        try:
            self.extractors['color_hist'] = ColorHistogramExtractor()
            self.logger.info("Initialized Color Histogram extractor")
        except Exception as e:
            self.logger.error(f"Failed to initialize Color Histogram extractor: {e}")
        
        try:
            self.extractors['orb'] = ORBExtractor()
            self.logger.info("Initialized ORB extractor")
        except Exception as e:
            self.logger.error(f"Failed to initialize ORB extractor: {e}")
        
        # Vision Transformer variants (if available)
        if TORCH_AVAILABLE:
            try:
                self.extractors['vit_simple'] = VisionTransformerExtractor(variant='simple')
                self.logger.info("Initialized simple ViT extractor")
            except Exception as e:
                self.logger.warning(f"Failed to initialize simple ViT extractor: {e}")
            
            try:
                self.extractors['vit_multi_layer'] = VisionTransformerExtractor(variant='multi_layer')
                self.logger.info("Initialized multi-layer ViT extractor")
            except Exception as e:
                self.logger.warning(f"Failed to initialize multi-layer ViT extractor: {e}")
            
            try:
                self.extractors['vit_multi_scale'] = VisionTransformerExtractor(variant='multi_scale')
                self.logger.info("Initialized multi-scale ViT extractor")
            except Exception as e:
                self.logger.warning(f"Failed to initialize multi-scale ViT extractor: {e}")
        
        # Ensemble extractor (combining available extractors)
        if len(self.extractors) > 1:
            try:
                base_extractors = [self.extractors['color_hist']]
                if 'orb' in self.extractors:
                    base_extractors.append(self.extractors['orb'])
                if 'vit_simple' in self.extractors:
                    base_extractors.append(self.extractors['vit_simple'])
                
                self.extractors['ensemble'] = EnsembleExtractor(base_extractors)
                self.logger.info("Initialized ensemble extractor")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ensemble extractor: {e}")
        
        self.logger.info(f"Successfully initialized {len(self.extractors)} feature extractors")
    
    def _generate_content_hash(self, image_data: bytes) -> str:
        """Generate content-based hash for image data."""
        return hashlib.sha256(image_data).hexdigest()[:16]
    
    def _validate_image_data(self, image_data: bytes) -> bool:
        """Validate that image data is readable."""
        try:
            # Try to decode with PIL first
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            return True
        except Exception as pil_err:
            try:
                # Try with OpenCV as fallback
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img is not None and img.size > 0
            except Exception as cv_err:
                self.logger.warning(f"Image validation failed - PIL: {pil_err}, OpenCV: {cv_err}")
                return False
    
    async def add_tile(
        self, 
        image_data: bytes, 
        sku: str,
        model_name: str,
        collection_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: str = "image/jpeg"
    ) -> str:
        """Add a new tile to the database.
        
        Args:
            image_data: Raw image data as bytes
            sku: Unique SKU for the tile
            model_name: Model name of the tile
            collection_name: Collection name the tile belongs to
            description: Optional description of the tile
            metadata: Additional metadata as key-value pairs
            content_type: MIME type of the image (default: image/jpeg)
            
        Returns:
            str: The tile ID that was assigned
            
        Raises:
            ValueError: If image data is invalid or SKU already exists
            RuntimeError: If tile storage fails
        """
        # Validate input
        if not image_data:
            raise ValueError("Image data cannot be empty")
            
        if len(image_data) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image size {len(image_data)} exceeds maximum size of {MAX_IMAGE_SIZE} bytes")
            
        # Check if SKU already exists
        existing_tile = await Tile.find_one(Tile.sku == sku)
        if existing_tile:
            raise ValueError(f"Tile with SKU '{sku}' already exists")
        
        # Validate image data
        if not self._validate_image_data(image_data):
            raise ValueError("Invalid image data: Could not read image with PIL or OpenCV")
            
        try:
            # Try to open the image to validate it
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify that it is an image
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Convert back to bytes for storage
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            image_data = img_byte_arr.getvalue()
            
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Invalid image data: {e}")
        
        # Create new tile document
        tile = Tile(
            sku=sku,
            model_name=model_name,
            collection_name=collection_name,
            image_data=image_data,
            content_type=content_type,
            description=description,
            metadata=metadata or {}
        )
        
        try:
            # Save to database
            await tile.save()
            tile_id = str(tile.id)
            
            # Cache features for the new tile
            await self._cache_tile_features(tile_id, image_data)
            
            # Manage cache size
            self._manage_cache_size()
            
            self.logger.info(f"Added new tile with ID: {tile_id} (SKU: {sku})")
            return tile_id
            
        except Exception as e:
            self.logger.error(f"Failed to save tile to database: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save tile: {e}")
    
    def _load_images_from_directory(self):
        """Load all images from the uploads directory. Fixed to avoid recursion."""
        if not self.upload_dir.exists():
            self.logger.warning(f"Upload directory does not exist: {self.upload_dir}")
            return
        
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = [
                f for f in self.upload_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            loaded_count = 0
            for img_path in image_files:
                try:
                    # Validate path is within upload directory (security fix)
                    if not str(img_path.resolve()).startswith(str(self.upload_dir.resolve())):
                        self.logger.warning(f"Skipping file outside upload directory: {img_path}")
                        continue
                    
                    # Read image data
                    with open(img_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Create metadata
                    metadata = {
                        'filename': img_path.name,
                        'file_size': len(image_data),
                        'source': 'directory_load'
                    }
                    
                    # Generate tile ID and add directly to avoid recursion
                    tile_id = self._generate_content_hash(image_data)
                    
                    # Store tile data directly (avoiding add_tile to prevent recursion)
                    tile_data = {
                        'id': tile_id,
                        'image_data': base64.b64encode(image_data).decode('utf-8'),
                        'metadata': metadata,
                        'created_at': time.time()
                    }
                    
                    self.tiles[tile_id] = tile_data
                    self._cache_tile_features(tile_id, image_data)
                    
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error loading image {img_path}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} images from directory")
            
        except Exception as e:
            self.logger.error(f"Error scanning upload directory: {e}")
    
    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries if needed."""
        if len(self.feature_cache) > self.max_cache_size:
            # Get oldest tiles by creation time
            tiles_by_age = sorted(
                self.tiles.items(),
                key=lambda x: x[1].get('created_at', 0)
            )
            
            # Remove oldest tiles from cache
            to_remove = len(self.feature_cache) - self.max_cache_size
            for i in range(to_remove):
                tile_id = tiles_by_age[i][0]
                if tile_id in self.feature_cache:
                    del self.feature_cache[tile_id]
                    self.logger.debug(f"Removed tile {tile_id} from cache")
    
    def _cache_tile_features(self, tile_id: str, image_data: bytes):
        """Pre-compute and cache features for all extraction methods."""
        try:
            self.feature_cache[tile_id] = {}
            
            for method in FEATURE_DIMENSIONS.keys():
                try:
                    features = self._extract_features_from_bytes(image_data, method)
                    self.feature_cache[tile_id][method] = features
                except Exception as e:
                    self.logger.warning(f"Failed to extract {method} features for tile {tile_id}: {e}")
                    # Store empty array to prevent re-computation attempts
                    self.feature_cache[tile_id][method] = np.array([])
                    
        except Exception as e:
            self.logger.error(f"Error caching features for tile {tile_id}: {e}")
    
    def _extract_features_from_bytes(self, image_data: bytes, method: str) -> np.ndarray:
        """Extract features from image bytes using enhanced extractors.
        
        Args:
            image_data: Raw image bytes
            method: Feature extraction method
            
        Returns:
            np.ndarray: Feature vector
            
        Raises:
            ValueError: If method is unsupported or image is invalid
        """
        if method not in self.extractors:
            # Try backward compatibility mapping
            legacy_mapping = {
                'vit': 'vit_simple',
                'color_hist': 'color_hist',
                'orb': 'orb'
            }
            if method in legacy_mapping and legacy_mapping[method] in self.extractors:
                method = legacy_mapping[method]
            else:
                raise ValueError(f"Unsupported feature extraction method: {method}")
        
        try:
            # Decode image from bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None or img.size == 0:
                raise ValueError("Could not decode image data")
            
            # Use the appropriate extractor
            extractor = self.extractors[method]
            features = extractor.extract(img)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting {method} features: {e}")
            raise
    
    def _extract_features_from_image(self, img: np.ndarray, method: str) -> np.ndarray:
        """Extract features from OpenCV image array.
        
        Returns feature vector of consistent, predetermined size.
        """
        target_dim = FEATURE_DIMENSIONS.get(method)
        if not target_dim:
            raise ValueError(f"Unsupported feature extraction method: {method}")
            
        if method == 'color_hist':
            return self._extract_color_histogram(img, target_dim)
        elif method == 'orb':
            return self._extract_orb_features(img, target_dim)
        elif method == 'vit':
            return self._extract_vit_features(img, target_dim)
        elif method == 'sift':
            return self._extract_sift_features(img, target_dim)
        elif method == 'kaze':
            return self._extract_kaze_features(img, target_dim)
        elif method == 'clip':
            return self._extract_clip_features(img, target_dim)
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")
    
    def _extract_color_histogram(self, img: np.ndarray, target_dim: int) -> np.ndarray:
        """Extract normalized color histogram features."""
        try:
            # Calculate 3D color histogram (8x8x8 = 512 bins)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Normalize and flatten
            features = cv2.normalize(hist, hist).flatten()
            
            # Ensure exact target dimension
            return self._ensure_feature_dimension(features, target_dim)
            
        except Exception as e:
            self.logger.error(f"Color histogram extraction failed: {e}")
            return np.zeros(target_dim)
    
    def _extract_orb_features(self, img: np.ndarray, target_dim: int) -> np.ndarray:
        """Extract ORB features using bag-of-words approach for consistent dimensionality."""
        try:
            # Convert to grayscale for ORB
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB keypoints and descriptors
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) == 0:
                self.logger.warning("No ORB features detected")
                return np.zeros(target_dim)
            
            # Build/use vocabulary for bag-of-words
            bow_features = self._compute_bow_features(descriptors, 'orb', target_dim)
            
            return self._ensure_feature_dimension(bow_features, target_dim)
            
        except Exception as e:
            self.logger.error(f"ORB feature extraction failed: {e}")
            return np.zeros(target_dim)
    
    def _compute_bow_features(self, descriptors: np.ndarray, method: str, target_dim: int) -> np.ndarray:
        """Compute bag-of-words features from descriptors. Fixed to use separate vocabularies."""
        try:
            # Get or build vocabulary for this specific method
            if self.vocabularies[method] is None:
                self._build_vocabulary(method)
            
            # If still no vocabulary, use descriptor statistics
            if self.vocabularies[method] is None:
                return self._compute_descriptor_statistics(descriptors, target_dim)
            
            # Compute distances to vocabulary words
            vocabulary = self.vocabularies[method]
            bow_histogram = np.zeros(len(vocabulary))
            
            for desc in descriptors:
                # Find closest vocabulary word
                distances = np.linalg.norm(vocabulary - desc, axis=1)
                closest_word = np.argmin(distances)
                bow_histogram[closest_word] += 1
            
            # Normalize
            if np.sum(bow_histogram) > 0:
                bow_histogram = bow_histogram / np.sum(bow_histogram)
            
            return self._ensure_feature_dimension(bow_histogram, target_dim)
            
        except Exception as e:
            self.logger.error(f"BoW computation failed for {method}: {e}")
            return np.zeros(target_dim, dtype=np.float32)
    
    def _build_vocabulary(self, method: str):
        """Build vocabulary from existing tiles for bag-of-words."""
        try:
            all_descriptors = []
            
            # Get the appropriate detector and vocabulary size
            if method == 'orb':
                detector = self.orb
                vocab_size = self.orb_vocab_size
            elif method == 'sift':
                detector = self.sift
                vocab_size = self.sift_vocab_size
            elif method == 'kaze':
                detector = self.kaze
                vocab_size = self.kaze_vocab_size
            else:
                self.logger.error(f"Unsupported method for vocabulary building: {method}")
                return
            
            # Collect descriptors from existing tiles
            for tile_id, tile_data in self.tiles.items():
                try:
                    # Safe base64 decoding
                    try:
                        image_data = base64.b64decode(tile_data['image_data'])
                    except Exception as e:
                        self.logger.warning(f"Failed to decode base64 for tile {tile_id}: {e}")
                        continue
                        
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        _, descriptors = detector.detectAndCompute(gray, None)
                        
                        if descriptors is not None and len(descriptors) > 0:
                            all_descriptors.append(descriptors)
                            
                except Exception as e:
                    self.logger.warning(f"Error processing tile {tile_id} for {method} vocabulary: {e}")
            
            if not all_descriptors:
                self.logger.warning(f"No descriptors found for {method} vocabulary")
                return
            
            # Combine all descriptors
            combined_descriptors = np.vstack(all_descriptors)
            
            # Use KMeans to create vocabulary
            if len(combined_descriptors) >= vocab_size:
                kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10)
                kmeans.fit(combined_descriptors)
                self.vocabularies[method] = kmeans.cluster_centers_
                self.logger.info(f"Built {method} vocabulary with {len(self.vocabularies[method])} words")
            else:
                self.logger.warning(f"Insufficient descriptors ({len(combined_descriptors)}) for vocabulary")
                
        except Exception as e:
            self.logger.error(f"Error building {method} vocabulary: {e}")
    
    def _compute_descriptor_statistics(self, descriptors: np.ndarray, target_dim: int) -> np.ndarray:
        """Compute statistical features from descriptors when vocabulary is unavailable."""
        features = []
        desc_dim = min(descriptors.shape[1], target_dim // 4)
        
        for i in range(desc_dim):
            col = descriptors[:, i]
            features.extend([
                np.mean(col),
                np.std(col),
                np.median(col),
                np.percentile(col, 75) - np.percentile(col, 25)  # IQR
            ])
        
        # Pad to target dimension
        features = np.array(features[:target_dim])
        if len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)), 'constant')
        
        return features.astype(np.float32)
    
    def _extract_vit_features(self, img: np.ndarray, target_dim: int) -> np.ndarray:
        """Extract VIT-like features using patch-based analysis."""
        try:
            # Convert to RGB if grayscale
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Resize to standard size for patch extraction
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values
            img_norm = img.astype(np.float32) / 255.0
            
            # Split into patches (simplified approach)
            patches = []
            patch_size = 16
            for y in range(0, img.shape[0], patch_size):
                for x in range(0, img.shape[1], patch_size):
                    if y + patch_size <= img.shape[0] and x + patch_size <= img.shape[1]:
                        patch = img_norm[y:y+patch_size, x:x+patch_size]
                        patches.append(patch.flatten())
            
            # Combine patches
            if not patches:
                return np.zeros(target_dim, dtype=np.float32)
            patches = np.array(patches)
            
            # Calculate patch statistics
            mean_patch = np.mean(patches, axis=0)
            std_patch = np.std(patches, axis=0)
            max_patch = np.max(patches, axis=0)
            min_patch = np.min(patches, axis=0)
            
            # Combine statistics
            features = np.concatenate([mean_patch, std_patch, max_patch, min_patch])
            
            # Ensure consistent dimension
            features = self._ensure_feature_dimension(features, target_dim)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting VIT features: {e}")
            return np.zeros(target_dim, dtype=np.float32)
            
    def _extract_sift_features(self, img: np.ndarray, target_dim: int) -> np.ndarray:
        """Extract SIFT features using bag-of-words approach for consistent dimensionality."""
        try:
            # Convert to grayscale for SIFT
            if len(img.shape) > 2 and img.shape[2] > 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            # Handle images with no keypoints
            if descriptors is None or len(keypoints) == 0:
                self.logger.debug("No SIFT keypoints found, returning zero vector")
                return np.zeros(target_dim, dtype=np.float32)
                
            # SIFT descriptors are already 128-dim per keypoint
            # Use bag-of-words approach similar to ORB for fixed dimensions
            features = self._compute_bow_features(descriptors, 'sift', target_dim)
            
            # Ensure exact dimension
            features = self._ensure_feature_dimension(features, target_dim)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting SIFT features: {e}")
            return np.zeros(target_dim, dtype=np.float32)
            
    def _extract_kaze_features(self, img: np.ndarray, target_dim: int) -> np.ndarray:
        """Extract KAZE features using bag-of-words approach for consistent dimensionality."""
        try:
            # Convert to grayscale for KAZE
            if len(img.shape) > 2 and img.shape[2] > 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.kaze.detectAndCompute(gray, None)
            
            # Handle images with no keypoints
            if descriptors is None or len(keypoints) == 0:
                self.logger.debug("No KAZE keypoints found, returning zero vector")
                return np.zeros(target_dim, dtype=np.float32)
                
            # KAZE descriptors are 64-dim per keypoint (or 128 if extended=True)
            # Use bag-of-words approach similar to ORB for fixed dimensions
            features = self._compute_bow_features(descriptors, 'kaze', target_dim)
            
            # Ensure exact dimension
            features = self._ensure_feature_dimension(features, target_dim)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting KAZE features: {e}")
            return np.zeros(target_dim, dtype=np.float32)
    
    def _extract_clip_features(self, img: np.ndarray, target_dim: int) -> np.ndarray:
        """Extract CLIP features using OpenAI's CLIP model."""
        try:
            # Check if CLIP model is available
            if self.clip_model is None:
                self.logger.error("CLIP model not available, returning zero vector")
                return np.zeros(target_dim, dtype=np.float32)
            
            # Convert OpenCV BGR to RGB
            if len(img.shape) > 2 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Handle grayscale by converting to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
            
            # Convert to PIL image for CLIP preprocessing
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess image for CLIP
            with torch.no_grad():
                # Apply CLIP preprocessing and move to device
                image_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.clip_device)
                
                # Get image features
                image_features = self.clip_model.encode_image(image_input)
                
                # Move back to CPU, normalize, and convert to numpy
                image_features = image_features.cpu().numpy()[0]
                
                # Normalize features
                norm = np.linalg.norm(image_features)
                if norm > 0:
                    image_features = image_features / norm
            
            # Ensure exact dimension
            features = self._ensure_feature_dimension(image_features, target_dim)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting CLIP features: {e}")
            return np.zeros(target_dim, dtype=np.float32)
    
    def _ensure_feature_dimension(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Ensure feature vector has exact target dimension."""
        if len(features) == target_dim:
            return features
        elif len(features) < target_dim:
            # Pad with zeros
            return np.pad(features, (0, target_dim - len(features)), 'constant')
        else:
            # Truncate
            return features[:target_dim]
    
    def get_tile_image_data(self, tile_id: str) -> Optional[dict]:
        """Get the base64-encoded image data for a tile.
        
        Args:
            tile_id: The ID of the tile
            
        Returns:
            dict containing 'data' (base64 string) and 'content_type', or None if not found
        """
        try:
            tile = self.tiles.get(tile_id)
            if not tile:
                return None
                
            # Image data should always be stored in base64 format
            if 'image_data' not in tile:
                self.logger.error(f"No image data found for tile {tile_id}")
                return None
            
            # Determine content type from metadata or filename
            content_type = 'image/jpeg'  # default
            if 'metadata' in tile:
                if 'content_type' in tile['metadata']:
                    content_type = tile['metadata']['content_type']
                elif 'filename' in tile['metadata']:
                    filename = tile['metadata']['filename'].lower()
                    if filename.endswith('.png'):
                        content_type = 'image/png'
                    elif filename.endswith('.webp'):
                        content_type = 'image/webp'
                    elif filename.endswith(('.jpg', '.jpeg')):
                        content_type = 'image/jpeg'
            
            return {
                'data': tile['image_data'],
                'content_type': content_type
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving image data for tile {tile_id}: {e}")
            return None
    
    def _get_cached_features(self, tile_id: str, method: str) -> Optional[np.ndarray]:
        """Get cached features for a tile and method.
        
        Args:
            tile_id: ID of the tile
            method: Feature extraction method
            
        Returns:
            Cached feature vector or None if not available
        """
        if tile_id in self.feature_cache and method in self.feature_cache[tile_id]:
            features = self.feature_cache[tile_id][method]
            # Return None for empty arrays (failed extractions)
            return features if features.size > 0 else None
        return None
    
    def _calculate_similarity(self, feat1: np.ndarray, feat2: np.ndarray, method: str = 'cosine') -> float:
        """Calculate similarity between two feature vectors using enhanced calculator.
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector  
            method: Similarity calculation method
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Validate inputs
            if feat1.size == 0 or feat2.size == 0:
                return 0.0
            
            # Ensure same dimensions
            if feat1.shape != feat2.shape:
                self.logger.warning(f"Feature dimension mismatch: {feat1.shape} vs {feat2.shape}")
                return 0.0
            
            # Map method to SimilarityMetric enum
            metric_mapping = {
                'cosine': SimilarityMetric.COSINE,
                'euclidean': SimilarityMetric.EUCLIDEAN,
                'manhattan': SimilarityMetric.MANHATTAN,
                'pearson': SimilarityMetric.PEARSON
            }
            
            metric = metric_mapping.get(method, SimilarityMetric.COSINE)
            
            # Use the enhanced similarity calculator
            similarity = self.similarity_calculator.calculate_similarity(feat1, feat2, metric)
            
            return float(similarity)
                
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_tiles(self, query_image_data: bytes, top_k: int = 5, method: str = 'color_hist', threshold: float = 0.0) -> List[dict]:
        """Find similar tiles to the query image using cached features.
        
        Args:
            query_image_data: Raw image data as bytes
            top_k: Maximum number of matches to return
            method: Feature extraction method to use ('color_hist', 'orb', 'vit')
            threshold: Minimum similarity score (0-1) for a match to be included
            
        Returns:
            List of matching tiles with similarity scores, sorted by score (highest first)
            
        Raises:
            ValueError: If method is unsupported
        """
        if not self.tiles:
            self.logger.warning("No tiles available for matching")
            return []
        
        if method not in self.extractors:
            available_methods = list(self.extractors.keys())
            raise ValueError(f"Unsupported method: {method}. Available: {available_methods}")
            
        try:
            # Extract features from query image
            query_features = self._extract_features_from_bytes(query_image_data, method)
            
            if query_features.size == 0:
                self.logger.error("Failed to extract features from query image")
                return []
            
            # Compare with all tiles using cached features
            similarities = []
            
            for tile_id in self.tiles.keys():
                try:
                    # Get cached features
                    tile_features = self._get_cached_features(tile_id, method)
                    
                    if tile_features is not None:
                        score = self._calculate_similarity(query_features, tile_features, 'cosine')
                        
                        if score >= threshold:
                            similarities.append((tile_id, float(score)))
                    else:
                        self.logger.debug(f"No cached {method} features for tile {tile_id}")
                        
                except Exception as e:
                    self.logger.warning(f"Error comparing with tile {tile_id}: {e}")
                    continue
            
            # Sort by score (highest first) and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:top_k]
            
            # Prepare results
            results = []
            for tile_id, score in similarities:
                try:
                    tile_data = self.tiles[tile_id].copy()
                    # Remove raw image data from response to save bandwidth
                    if 'image_data' in tile_data:
                        del tile_data['image_data']
                    
                    results.append({
                        'tile_id': tile_id,
                        'similarity': score,
                        'metadata': tile_data.get('metadata', {}),
                        'created_at': tile_data.get('created_at')
                    })
                except Exception as e:
                    self.logger.warning(f"Error preparing result for tile {tile_id}: {e}")
                    continue
            
            self.logger.info(f"Found {len(results)} matches using {method} method")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in find_similar_tiles: {e}", exc_info=True)
            return []
    
    def search_tiles(self, query_image_data: bytes, **kwargs) -> List[dict]:
        """Convenience method for tile search - alias for find_similar_tiles."""
        return self.find_similar_tiles(query_image_data, **kwargs)
    
    def get_tile_count(self) -> int:
        """Get the total number of tiles in the service."""
        return len(self.tiles)
    
    
    def remove_tile(self, tile_id: str) -> bool:
        """Remove a tile from the service.
        
        Args:
            tile_id: ID of the tile to remove
            
        Returns:
            True if tile was removed, False if not found
        """
        try:
            if tile_id in self.tiles:
                del self.tiles[tile_id]
                
                # Also remove from feature cache
                if tile_id in self.feature_cache:
                    del self.feature_cache[tile_id]
                
                self.logger.info(f"Removed tile {tile_id}")
                return True
            else:
                self.logger.warning(f"Tile {tile_id} not found for removal")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing tile {tile_id}: {e}")
            return False
    
    def get_available_methods(self) -> List[str]:
        """Get list of available feature extraction methods."""
        return list(self.extractors.keys())
    
    def find_similar_tiles_ensemble(self, query_image_data: bytes, top_k: int = 5, 
                                  methods: Optional[List[str]] = None, threshold: float = 0.0) -> List[MatchResult]:
        """Find similar tiles using ensemble approach with multiple methods.
        
        Args:
            query_image_data: Raw image data as bytes
            top_k: Maximum number of matches to return
            methods: List of methods to use (if None, uses all available)
            threshold: Minimum similarity score for inclusion
            
        Returns:
            List of MatchResult objects with confidence scores
        """
        if not self.tiles:
            return []
        
        if methods is None:
            methods = [m for m in self.extractors.keys() if m != 'ensemble']
        
        # Filter to available methods
        methods = [m for m in methods if m in self.extractors]
        
        if not methods:
            self.logger.warning("No valid methods available for ensemble matching")
            return []
        
        try:
            # Get query features for each method
            query_features = {}
            for method in methods:
                try:
                    features = self._extract_features_from_bytes(query_image_data, method)
                    if features.size > 0:
                        query_features[method] = features
                except Exception as e:
                    self.logger.warning(f"Failed to extract {method} features from query: {e}")
            
            if not query_features:
                return []
            
            # Calculate similarities for each tile and method
            tile_scores = {}
            
            for tile_id in self.tiles.keys():
                method_similarities = []
                
                for method in query_features.keys():
                    tile_features = self._get_cached_features(tile_id, method)
                    if tile_features is not None:
                        similarity, confidence = self.similarity_calculator.calculate_ensemble_similarity(
                            query_features[method], tile_features
                        )
                        method_similarities.append((similarity, confidence))
                
                if method_similarities:
                    # Combine similarities across methods
                    similarities = [s[0] for s in method_similarities]
                    confidences = [s[1] for s in method_similarities]
                    
                    # Weighted average based on confidence
                    weights = np.array(confidences)
                    weights = weights / (np.sum(weights) + 1e-8)
                    
                    combined_similarity = np.sum(np.array(similarities) * weights)
                    combined_confidence = np.mean(confidences)
                    
                    if combined_similarity >= threshold:
                        tile_scores[tile_id] = (combined_similarity, combined_confidence)
            
            # Sort by similarity and take top_k
            sorted_tiles = sorted(tile_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
            
            # Prepare results
            results = []
            for tile_id, (similarity, confidence) in sorted_tiles:
                tile_data = self.tiles[tile_id]
                result = MatchResult(
                    tile_id=tile_id,
                    similarity=float(similarity),
                    confidence=float(confidence),
                    method='ensemble',
                    metadata=tile_data.get('metadata', {})
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ensemble matching: {e}")
            return []
    
    async def find_similar_tiles_async(self, query_image_data: bytes, top_k: int = 5, 
                                     method: str = 'color_hist', threshold: float = 0.0) -> List[dict]:
        """Async version of similarity search for better performance."""
        
        def search_worker():
            return self.find_similar_tiles(query_image_data, top_k, method, threshold)
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, search_worker)
        
        return result
    
    def batch_find_similar_tiles(self, query_images: List[bytes], top_k: int = 5, 
                               method: str = 'color_hist', threshold: float = 0.0) -> List[List[dict]]:
        """Batch process multiple query images for efficiency."""
        results = []
        
        for query_image_data in query_images:
            try:
                matches = self.find_similar_tiles(query_image_data, top_k, method, threshold)
                results.append(matches)
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                results.append([])
        
        return results
    
    def test_similarity_metrics(self, query_image_data: bytes, tile_id: str) -> Dict[str, float]:
        """Test different similarity metrics against a specific tile.
        
        Args:
            query_image_data: Query image as bytes
            tile_id: Target tile ID to compare against
            
        Returns:
            Dictionary mapping metric names to similarity scores
        """
        if tile_id not in self.tiles:
            return {}
        
        results = {}
        
        # Use the first available method for feature extraction
        method = next(iter(self.extractors.keys()), 'color_hist')
        
        try:
            # Extract query features
            query_features = self._extract_features_from_bytes(query_image_data, method)
            tile_features = self._get_cached_features(tile_id, method)
            
            if tile_features is None or query_features.size == 0:
                return {}
            
            # Test all similarity metrics
            for metric in SimilarityMetric:
                try:
                    similarity = self.similarity_calculator.calculate_similarity(
                        query_features, tile_features, metric
                    )
                    results[metric.value] = float(similarity)
                except Exception as e:
                    self.logger.warning(f"Error testing {metric.value}: {e}")
                    results[metric.value] = 0.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity metrics test: {e}")
            return {}

# Global instance of the matching service
_matching_service: Optional[TileMatchingService] = None
_matching_service_lock = threading.Lock()

async def get_matching_service() -> TileMatchingService:
    """
    Dependency function to get the matching service instance (singleton).
    Initializes the service and starts background tasks on first call.
    """
    global _matching_service
    if _matching_service is None:
        with _matching_service_lock:
            # Double-check lock to prevent race conditions
            if _matching_service is None:
                service = TileMatchingService()
                # Start the preloading task from within an async context
                service._preload_task = asyncio.create_task(service._preload_feature_cache())
                _matching_service = service

    return _matching_service
