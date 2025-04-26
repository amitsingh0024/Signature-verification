import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import uvicorn
from models.siamese_network import SiameseNetwork

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(
    title="Signature Verification API",
    description="API for verifying signatures using a Siamese Network",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files with absolute path
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize model and device
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = None

def load_model():
    global model, transform
    try:
        # Load the model
        model = SiameseNetwork()
        model_path = os.path.join(project_root, "models", "signature_verification_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the image for the model"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/verify")
async def verify_signatures(
    reference_signature: UploadFile = File(...),
    test_signature: UploadFile = File(...)
):
    try:
        # Read and verify images
        ref_image = Image.open(io.BytesIO(await reference_signature.read()))
        test_image = Image.open(io.BytesIO(await test_signature.read()))
        
        # Preprocess images
        ref_tensor = preprocess_image(ref_image)
        test_tensor = preprocess_image(test_image)
        
        # Move tensors to device
        ref_tensor = ref_tensor.to(device)
        test_tensor = test_tensor.to(device)
        
        # Get model prediction
        with torch.no_grad():
            output1, output2 = model(ref_tensor, test_tensor)
            similarity = torch.cosine_similarity(output1, output2)
            similarity_score = similarity.item()
            
            # Convert similarity score to prediction and confidence
            prediction = "Genuine" if similarity_score > 0.5 else "Forged"
            confidence = abs(similarity_score - 0.5) * 2  # Scale to 0-1 range
        
        return {
            "similarity_score": similarity_score,
            "prediction": prediction,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 