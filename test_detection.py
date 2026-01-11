"""
Test the detection endpoint with a sample image
"""

import requests
import json
from pathlib import Path
import base64
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple green image (simulating a leaf)
    img = Image.new('RGB', (640, 480), color='green')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_detection_endpoint():
    """Test the detection endpoint"""
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare the request
        files = {
            'file': ('test_leaf.jpg', test_image, 'image/jpeg')
        }
        
        data = {
            'confidence': 0.7  # High confidence level
        }
        
        print("🔍 Testing detection endpoint...")
        print(f"Confidence level: {data['confidence']}")
        
        # Send request
        response = requests.post(
            "http://localhost:8000/detect",
            files=files,
            data=data,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection successful!")
            print(f"Predictions count: {result.get('predictions_count', 0)}")
            print(f"Health status: {result.get('health_status', 'Unknown')}")
            
            if result.get('detailed_predictions'):
                print("\nDetailed predictions:")
                for pred in result['detailed_predictions']:
                    print(f"  - {pred['class']}: {pred['confidence']:.2%}")
            else:
                print("No diseases detected - healthy leaf!")
                
            return True
        else:
            print(f"❌ Detection failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run the detection test"""
    print("🧪 Testing Maize Disease Detection Endpoint")
    print("=" * 50)
    
    success = test_detection_endpoint()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Detection endpoint is working correctly!")
        print("✅ Backend is ready for frontend connection")
    else:
        print("❌ Detection endpoint has issues")
        print("🔧 Check server logs for more details")

if __name__ == "__main__":
    main()