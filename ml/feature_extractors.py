import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel, CLIPProcessor, CLIPModel
import torch.nn.functional as F
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureExtractorConfig:
    method: str  # 'sift', 'orb', 'kaze', 'vit', 'clip'
    n_features: int = 1000
    match_threshold: float = 0.7

class BaseFeatureExtractor:
    def __init__(self, config: FeatureExtractorConfig):
        self.config = config
    
    def extract(self, image: np.ndarray) -> Dict:
        raise NotImplementedError
    
    def match(self, features1: Dict, features2: Dict) -> float:
        raise NotImplementedError

class SIFTExtractor(BaseFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self.sift = cv2.SIFT_create(nfeatures=self.config.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def extract(self, image: np.ndarray) -> Dict:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
    
    def match(self, features1: Dict, features2: Dict) -> float:
        if features1['descriptors'] is None or features2['descriptors'] is None:
            return 0.0
            
        matches = self.matcher.knnMatch(
            features1['descriptors'], 
            features2['descriptors'], 
            k=2
        )
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.config.match_threshold * n.distance:
                good_matches.append(m)
        
        if len(features1['descriptors']) == 0 or len(features2['descriptors']) == 0:
            return 0.0
            
        return len(good_matches) / min(len(features1['descriptors']), len(features2['descriptors']))

class ORBExtractor(BaseFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self.orb = cv2.ORB_create(nfeatures=self.config.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def extract(self, image: np.ndarray) -> Dict:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
    
    def match(self, features1: Dict, features2: Dict) -> float:
        if features1['descriptors'] is None or features2['descriptors'] is None:
            return 0.0
            
        matches = self.matcher.knnMatch(
            features1['descriptors'], 
            features2['descriptors'], 
            k=2
        )
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.config.match_threshold * n.distance:
                good_matches.append(m)
        
        if len(features1['descriptors']) == 0 or len(features2['descriptors']) == 0:
            return 0.0
            
        return len(good_matches) / min(len(features1['descriptors']), len(features2['descriptors']))

class KAZEExtractor(BaseFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self.kaze = cv2.KAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def extract(self, image: np.ndarray) -> Dict:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.kaze.detectAndCompute(gray, None)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
    
    def match(self, features1: Dict, features2: Dict) -> float:
        if features1['descriptors'] is None or features2['descriptors'] is None:
            return 0.0
            
        matches = self.matcher.knnMatch(
            features1['descriptors'], 
            features2['descriptors'], 
            k=2
        )
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.config.match_threshold * n.distance:
                good_matches.append(m)
        
        if len(features1['descriptors']) == 0 or len(features2['descriptors']) == 0:
            return 0.0
            
        return len(good_matches) / min(len(features1['descriptors']), len(features2['descriptors']))

class ViTExtractor(BaseFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained ViT model
        self.processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)
        self.model.eval()
    
    def extract(self, image: np.ndarray) -> Dict:
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Single channel
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # Already RGB
            image_rgb = image
        
        # Preprocess and extract features
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token representation as the image embedding
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return {
            'features': features,
            'tensor': torch.tensor(features).unsqueeze(0)
        }
    
    def match(self, features1: Dict, features2: Dict) -> float:
        # Calculate cosine similarity between feature vectors
        if 'tensor' not in features1 or 'tensor' not in features2:
            return 0.0
            
        # Ensure tensors are on the same device
        tensor1 = features1['tensor'].to(self.device)
        tensor2 = features2['tensor'].to(self.device)
        
        # Normalize the feature vectors
        tensor1 = F.normalize(tensor1, p=2, dim=1)
        tensor2 = F.normalize(tensor2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(tensor1, tensor2)
        return float(similarity[0])


class CLIPExtractor(BaseFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained CLIP model
        model_name = "openai/clip-vit-base-patch32"
        logger.info(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def extract(self, image: np.ndarray) -> Dict:
        # Convert to RGB if needed (CLIP expects RGB format)
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Single channel
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:  # BGR (OpenCV default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:  # Already RGB
            image_rgb = image
            
        # Convert to PIL Image which is expected by the CLIP processor
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess and extract features
        inputs = self.processor(images=pil_image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # Extract the image features
        features = outputs.cpu().numpy()[0]
        return {
            'features': features,
            'tensor': torch.tensor(features).unsqueeze(0)
        }
    
    def match(self, features1: Dict, features2: Dict) -> float:
        # Calculate cosine similarity between feature vectors
        if 'tensor' not in features1 or 'tensor' not in features2:
            return 0.0
            
        # Ensure tensors are on the same device
        tensor1 = features1['tensor'].to(self.device)
        tensor2 = features2['tensor'].to(self.device)
        
        # Normalize the feature vectors
        tensor1 = F.normalize(tensor1, p=2, dim=1)
        tensor2 = F.normalize(tensor2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(tensor1, tensor2)
        return float(similarity[0])

def get_feature_extractor(method: str, **kwargs) -> BaseFeatureExtractor:
    """Factory function to get the appropriate feature extractor"""
    config = FeatureExtractorConfig(method=method, **kwargs)
    
    if method == 'sift':
        return SIFTExtractor(config)
    elif method == 'orb':
        return ORBExtractor(config)
    elif method == 'kaze':
        return KAZEExtractor(config)
    elif method == 'vit':
        return ViTExtractor(config)
    elif method == 'clip':
        return CLIPExtractor(config)
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")
