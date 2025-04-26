import torch
import os
import sys
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.siamese_network import SiameseNetwork, SignatureDataset
from config.config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, PATHS, TRANSFORM, DEVICE

def test_model(model_path):
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
        mode='test'
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

def generate_model_report():
    # Create report content
    report = []
    report.append("=" * 80)
    report.append("MODEL REPORT")
    report.append("=" * 80)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Model Architecture
    report.append("MODEL ARCHITECTURE")
    report.append("-" * 40)
    model = SiameseNetwork()
    report.append(f"Model Type: Siamese Network with ResNet18 backbone")
    report.append(f"Embedding Size: {MODEL_CONFIG['embedding_size']}")
    report.append(f"Margin (Contrastive Loss): {MODEL_CONFIG['margin']}")
    report.append(f"Input Size: {MODEL_CONFIG['input_size']}")
    report.append(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    report.append(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    report.append(f"Device: {DEVICE}\n")

    # Training Configuration
    report.append("TRAINING CONFIGURATION")
    report.append("-" * 40)
    report.append(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
    report.append(f"Number of Epochs: {TRAIN_CONFIG['num_epochs']}")
    report.append(f"Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    report.append(f"Early Stopping Patience: {TRAIN_CONFIG['early_stopping_patience']}\n")

    # Data Configuration
    report.append("DATA CONFIGURATION")
    report.append("-" * 40)
    report.append(f"Training CSV: {DATA_CONFIG['train_csv']}")
    report.append(f"Test CSV: {DATA_CONFIG['test_csv']}")
    report.append(f"Root Directory: {DATA_CONFIG['root_dir']}\n")

    # Image Preprocessing
    report.append("IMAGE PREPROCESSING")
    report.append("-" * 40)
    report.append(f"Resize: {TRANSFORM['train']['resize']}")
    report.append(f"Mean Normalization: {TRANSFORM['train']['mean']}")
    report.append(f"Standard Deviation: {TRANSFORM['train']['std']}\n")

    # Model Paths
    report.append("MODEL PATHS")
    report.append("-" * 40)
    report.append(f"Model Save Path: {PATHS['model_save']}")
    report.append(f"Checkpoints Directory: {PATHS['checkpoints']}")
    report.append(f"Results Directory: {PATHS['results']}")
    report.append(f"Tensorboard Directory: {PATHS['tensorboard']}\n")

    # Model Performance
    report.append("MODEL PERFORMANCE")
    report.append("-" * 40)
    if os.path.exists(PATHS['model_save']):
        try:
            metrics = test_model(PATHS['model_save'])
            report.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"Precision: {metrics['precision']:.4f}")
            report.append(f"Recall: {metrics['recall']:.4f}")
            report.append(f"F1 Score: {metrics['f1_score']:.4f}")
            report.append("\nConfusion Matrix:")
            report.append(f"True Positives: {metrics['true_positives']}")
            report.append(f"True Negatives: {metrics['true_negatives']}")
            report.append(f"False Positives: {metrics['false_positives']}")
            report.append(f"False Negatives: {metrics['false_negatives']}")
            report.append(f"\nTotal Test Samples: {metrics['total_samples']}")
        except Exception as e:
            report.append(f"Error testing model: {str(e)}")
    else:
        report.append("Model file not found. Performance metrics not available.")

    # Save the report
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', f'model_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Model report generated successfully at: {report_path}")

if __name__ == "__main__":
    generate_model_report() 