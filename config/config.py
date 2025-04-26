import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
MODEL_CONFIG = {
    'input_size': (224, 224),
    'embedding_size': 128,
    'margin': 2.0
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.0001,
    'early_stopping_patience': 5
}

# Data configuration
DATA_CONFIG = {
    'train_csv': 'dataset/sign_data/train_data.csv',
    'test_csv': 'dataset/sign_data/test_data.csv',
    'root_dir': 'dataset/sign_data'
}

# Paths
PATHS = {
    'model_save': 'models/signature_verification_model.pth',
    'checkpoints': 'models/checkpoints',
    'results': 'results',
    'tensorboard': 'runs'
}

# Image preprocessing
TRANSFORM = {
    'train': {
        'resize': (224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
} 