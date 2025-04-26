import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def load_model(model_path, model_class, device):
    """Load a trained model"""
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transforms(config):
    """Get image transformations based on config"""
    return transforms.Compose([
        transforms.Resize(config['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

def preprocess_image(image_path, transform, device):
    """Load and preprocess an image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension

def save_plot(fig, filename, directory='results'):
    """Save a matplotlib figure"""
    os.makedirs(directory, exist_ok=True)
    fig.savefig(os.path.join(directory, filename))
    plt.close(fig)

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        'models',
        'data',
        'utils',
        'results',
        'config',
        'models/checkpoints',
        'results/plots',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('results/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__) 