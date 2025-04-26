import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import sys
import torchvision.transforms as transforms

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, TRANSFORM

class SignatureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        print(f"Loading dataset from: {csv_file}")
        print(f"Root directory: {root_dir}")
        print(f"Mode: {mode}")
        
        # Read CSV file
        self.data = pd.read_csv(csv_file, header=None)
        print(f"Loaded {len(self.data)} samples")
        print("First few rows of data:")
        print(self.data.head())
        
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize(TRANSFORM['train']['resize']),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM['train']['mean'], 
                              std=TRANSFORM['train']['std'])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Get the row as a Series
            row = self.data.iloc[idx]
            
            # Convert to string and strip any whitespace
            img1_rel_path = str(row[0]).strip()
            img2_rel_path = str(row[1]).strip()
            label = float(row[2])
            
            print(f"\nProcessing index {idx}:")
            print(f"Image 1 path: {img1_rel_path}")
            print(f"Image 2 path: {img2_rel_path}")
            print(f"Label: {label}")

            # Construct full paths based on mode
            img1_path = os.path.join(self.root_dir, self.mode, img1_rel_path)
            img2_path = os.path.join(self.root_dir, self.mode, img2_rel_path)

            print(f"Full path 1: {img1_path}")
            print(f"Full path 2: {img2_path}")

            # Check if files exist
            if not os.path.exists(img1_path):
                raise FileNotFoundError(f"Image not found: {img1_path}")
            if not os.path.exists(img2_path):
                raise FileNotFoundError(f"Image not found: {img2_path}")

            # Load and convert images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"\nError loading data at index {idx}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Row data: {self.data.iloc[idx] if idx < len(self.data) else 'Index out of range'}")
            # Return a dummy sample in case of error
            dummy_img = torch.zeros((3, *TRANSFORM['train']['resize']))
            return dummy_img, dummy_img, torch.tensor(0.0)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Using a pre-trained ResNet18 as the base network
        self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Add a new fully connected layer for the embedding
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, MODEL_CONFIG['embedding_size'])
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MODEL_CONFIG['margin']):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive 