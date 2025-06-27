import os
import cv2
import numpy as np
from ml.matching_service import TileMatchingService, load_image
from pathlib import Path
import time

def test_matching_service():
    """
    Test the TileMatchingService with sample images.
    Place your test images in the 'test_images' directory.
    """
    # Create test directories
    test_dir = Path("test_images")
    catalog_dir = test_dir / "catalog"
    query_dir = test_dir / "queries"
    
    # Create directories if they don't exist
    catalog_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(exist_ok=True)
    
    print(f"Test directories created at: {test_dir.absolute()}")
    print("Place your catalog images in the 'catalog' directory and query images in 'queries'")
    
    # Initialize the matching service
    print("\nInitializing TileMatchingService...")
    matching_service = TileMatchingService(methods=['sift', 'orb', 'kaze', 'vit'])
    
    # Add catalog images
    print("\nLoading catalog images...")
    catalog_images = list(catalog_dir.glob("*.jpg")) + list(catalog_dir.glob("*.png"))
    
    if not catalog_images:
        print(f"No catalog images found in {catalog_dir}. Please add some images and try again.")
        return
    
    for img_path in catalog_images:
        tile_id = img_path.stem
        print(f"Adding catalog image: {img_path.name}")
        matching_service.add_tile(
            tile_id=tile_id,
            image_path=str(img_path),
            metadata={
                'filename': img_path.name,
                'source': 'test_catalog'
            }
        )
    
    # Build FAISS index for faster search
    print("\nBuilding FAISS index for faster search...")
    matching_service.build_faiss_index()
    
    # Process query images
    query_images = list(query_dir.glob("*.jpg")) + list(query_dir.glob("*.png"))
    
    if not query_images:
        print(f"\nNo query images found in {query_dir}. Please add some images and try again.")
        return
    
    for query_path in query_images:
        print(f"\nProcessing query image: {query_path.name}")
        
        # Load query image
        query_image = load_image(str(query_path))
        if query_image is None:
            print(f"  Error: Could not load image {query_path}")
            continue
        
        # Time the matching process
        start_time = time.time()
        
        # Get matches
        results = matching_service.match_image(query_image, top_k=3)
        
        elapsed_time = time.time() - start_time
        
        print(f"  Found {len(results)} matches in {elapsed_time:.2f} seconds:")
        
        # Display top matches
        for i, result in enumerate(results, 1):
            print(f"  {i}. Tile ID: {result.tile_id}, Score: {result.score:.4f}, Method: {result.method}")
            if 'filename' in result.metadata:
                print(f"     File: {result.metadata['filename']}")

if __name__ == "__main__":
    test_matching_service()
