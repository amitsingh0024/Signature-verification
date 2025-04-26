import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models.siamese_network import SiameseNetwork, SignatureDataset
from utils.utils import load_model
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from config.config import MODEL_CONFIG, DATA_CONFIG, PATHS, TRANSFORM, DEVICE

def test_signature_pair(img1_path, img2_path, model_path='models/signature_verification_model.pth'):
    """
    Test a pair of signatures for verification
    
    Args:
        img1_path (str): Path to first signature image
        img2_path (str): Path to second signature image
        model_path (str): Path to the trained model weights
    
    Returns:
        tuple: (similarity_score, prediction)
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load the model
        model = load_model(model_path, SiameseNetwork, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

    try:
        # Load and preprocess the images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_tensor = transform(img1).unsqueeze(0).to(device)
        img2_tensor = transform(img2).unsqueeze(0).to(device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            output1, output2 = model(img1_tensor, img2_tensor)
            euclidean_distance = F.pairwise_distance(output1, output2)
            similarity_score = 1 - torch.sigmoid(euclidean_distance).item()

        # Determine prediction
        prediction = 'Match' if similarity_score > 0.5 else 'No Match'
        
        return similarity_score, prediction

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return None, None

def test_genuine_vs_forged(person_id, model_path='models/signature_verification_model.pth'):
    """
    Test a genuine signature against its forged version
    
    Args:
        person_id (str): ID of the person (e.g., '049')
        model_path (str): Path to the trained model weights
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct paths relative to project root
    base_dir = os.path.join(project_root, 'dataset', 'sign_data', 'test')
    genuine_dir = os.path.join(base_dir, person_id)
    forged_dir = os.path.join(base_dir, f"{person_id}_forg")
    
    print(f"Looking for signatures in:")
    print(f"Genuine directory: {genuine_dir}")
    print(f"Forged directory: {forged_dir}")
    
    if not os.path.exists(genuine_dir) or not os.path.exists(forged_dir):
        print(f"Error: Directories not found for person {person_id}")
        return
    
    # Get the first genuine and forged signatures
    genuine_files = sorted([f for f in os.listdir(genuine_dir) if f.lower().endswith('.png')])
    forged_files = sorted([f for f in os.listdir(forged_dir) if f.lower().endswith('.png')])
    
    if not genuine_files or not forged_files:
        print(f"Error: No signature files found for person {person_id}")
        print(f"Genuine files found: {genuine_files}")
        print(f"Forged files found: {forged_files}")
        return
    
    genuine_path = os.path.join(genuine_dir, genuine_files[0])
    forged_path = os.path.join(forged_dir, forged_files[0])
    
    print(f"\nTesting genuine vs forged signatures for person {person_id}:")
    print(f"Genuine signature: {genuine_path}")
    print(f"Forged signature: {forged_path}")
    
    similarity_score, prediction = test_signature_pair(genuine_path, forged_path, model_path)
    
    if similarity_score is not None:
        print(f"\nTest Results:")
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {'High' if similarity_score > 0.7 or similarity_score < 0.3 else 'Medium'}")

def test_model(model_path):
    """
    Test the model on the entire test dataset and calculate performance metrics
    
    Args:
        model_path (str): Path to the trained model weights
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Load the model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Create test dataset and dataloader
    test_dataset = SignatureDataset(
        csv_file=DATA_CONFIG['test_csv'],
        root_dir=DATA_CONFIG['root_dir'],
        transform=TRANSFORM['train'],
        mode='test'  # Specify test mode
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize metrics
    correct = 0
    total = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Test the model
    with torch.no_grad():
        for img1, img2, labels in tqdm(test_loader, desc="Testing"):
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            
            # Get model outputs
            output1, output2 = model(img1, img2)
            
            # Calculate distances
            distances = torch.pairwise_distance(output1, output2)
            
            # Convert distances to predictions (0 if distance < margin, 1 otherwise)
            predictions = (distances > MODEL_CONFIG['margin']).float()
            
            # Update metrics
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            # Update confusion matrix
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    # Calculate metrics
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test signature verification model')
    parser.add_argument('--mode', choices=['single', 'full'], default='full',
                      help='Test mode: single (test one pair) or full (test entire dataset)')
    parser.add_argument('--person_id', type=str, default='049',
                      help='Person ID for single test mode')
    parser.add_argument('--model_path', type=str, default=PATHS['model_save'],
                      help='Path to the model weights')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    if args.mode == 'single':
        test_genuine_vs_forged(args.person_id, args.model_path)
    else:
        metrics = test_model(args.model_path)
        print("\nModel Test Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"\nTotal Samples: {metrics['total_samples']}")

if __name__ == '__main__':
    main() 