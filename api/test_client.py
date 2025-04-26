import requests
import os
from PIL import Image
import io

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print("\nHealth Check Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_signature_verification(ref_path, test_path):
    """Test the signature verification endpoint"""
    try:
        # Check if files exist
        if not os.path.exists(ref_path) or not os.path.exists(test_path):
            print(f"Error: One or both files not found")
            print(f"Reference path: {ref_path}")
            print(f"Test path: {test_path}")
            return False

        # Open and verify the images
        try:
            Image.open(ref_path).verify()
            Image.open(test_path).verify()
        except Exception as e:
            print(f"Error: Invalid image file(s): {str(e)}")
            return False

        # Prepare the request
        url = "http://localhost:8000/verify"
        files = {
            "reference_signature": open(ref_path, "rb"),
            "test_signature": open(test_path, "rb")
        }

        # Send the request
        print("\nSending verification request...")
        response = requests.post(url, files=files)

        # Print results
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\nVerification Results:")
            print(f"Similarity Score: {result['similarity_score']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}")
        else:
            print(f"Error: {response.text}")

        return response.status_code == 200

    except Exception as e:
        print(f"Verification test failed: {str(e)}")
        return False

def main():
    # Test health check
    if not test_health_check():
        print("Health check failed. Make sure the API server is running.")
        return

    # Test signature verification
    # Using example paths - replace with your actual signature paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(project_root, "dataset", "sign_data", "test", "049", "01_049.png")
    test_path = os.path.join(project_root, "dataset", "sign_data", "test", "049_forg", "01_0114049.PNG")

    test_signature_verification(ref_path, test_path)

if __name__ == "__main__":
    main() 