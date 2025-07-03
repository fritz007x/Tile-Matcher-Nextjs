import requests
import os
import numpy as np
import cv2

# --- Create a dummy image for testing ---
def create_dummy_image(path="test_tile.png"):
    if not os.path.exists(path):
        # Create a simple 100x100 black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(path, image)
        print(f"Created dummy image: {path}")
    return path

# --- Test script configuration ---
BASE_URL = "http://127.0.0.1:8000"
UPLOAD_URL = f"{BASE_URL}/api/matching/upload"

# --- Main test function ---
def test_upload_endpoint():
    print("--- Testing Tile Upload Endpoint ---")
    
    # 1. Create a dummy image to upload
    image_path = create_dummy_image()
    
    # 2. Prepare the multipart/form-data payload
    files = {'file': (os.path.basename(image_path), open(image_path, 'rb'), 'image/png')}
    data = {
        'sku': 'TEST-SKU-001',
        'model_name': 'Test Model',
        'collection_name': 'Test Collection'
    }
    
    try:
        # 3. Send the POST request
        print(f"Sending request to {UPLOAD_URL}...")
        response = requests.post(UPLOAD_URL, files=files, data=data)
        
        # 4. Print the response from the server
        print(f"\nStatus Code: {response.status_code}")
        print("Response JSON:")
        try:
            print(response.json())
        except requests.exceptions.JSONDecodeError:
            print("Response is not in JSON format.")
            print("Response Text:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the FastAPI server is running.")

if __name__ == "__main__":
    test_upload_endpoint()
