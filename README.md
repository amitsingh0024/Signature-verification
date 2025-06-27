### Signature Verification System

A deep learning-based system for verifying handwritten signatures using Siamese Networks. This project uses a pre-trained ResNet18 backbone to learn signature embeddings and verify if two signatures belong to the same person.

## Features

- Siamese Network architecture with ResNet18 backbone
- Signature embedding generation
- Signature pair verification
- Performance metrics and reporting
- TensorBoard integration for training visualization
- Comprehensive model evaluation

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- PIL (Python Imaging Library)
- tensorboard
- tqdm

## Installation

1. Clone or download this repository
2. Install the required packages:
```bash
pip install torch torchvision pandas pillow tensorboard tqdm
```

## Project Structure

```
.
├── config/
│   └── config.py          # Configuration settings
├── dataset/
│   └── sign_data/         # Signature dataset
├── models/
│   └── siamese_network.py # Model architecture
├── scripts/
│   ├── generate_model_report.py  # Generate model report
│   ├── launch_tensorboard.py     # Launch TensorBoard
│   └── test_model.py             # Test model performance
└── reports/               # Generated reports
```

## Usage

### 1. Training the Model

Run the training script:
```bash
python scripts/train.py
```

### 2. Testing the Model

Test a single signature pair:
```bash
python scripts/test_model.py --mode single --person_id 049
```

Test the entire dataset:
```bash
python scripts/test_model.py --mode full
```

### 3. Generating Model Report

Generate a comprehensive model report:
```bash
python scripts/generate_model_report.py
```

The report will be saved in the `reports` directory with a timestamp.

### 4. Viewing Training Logs

Launch TensorBoard to view training metrics:
```bash
python scripts/launch_tensorboard.py
```

Then open http://localhost:6006 in your web browser.

## Model Architecture

The system uses a Siamese Network with:
- ResNet18 backbone (pre-trained)
- Custom embedding layer
- Contrastive loss function
- Margin-based similarity comparison

## Configuration

Key configuration parameters in `config/config.py`:
- Model parameters (embedding size, margin)
- Training parameters (batch size, epochs, learning rate)
- Data paths
- Image preprocessing settings

## Performance Metrics

The system provides:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Total test samples

## Troubleshooting

1. If you encounter "Model file not found" errors:
   - Make sure you've trained the model first
   - Check the model path in config.py

2. If TensorBoard doesn't show any data:
   - Verify that training logs exist in the runs directory
   - Check if the training script completed successfully

3. If you get data loading errors:
   - Verify your dataset structure matches the expected format
   - Check the CSV file paths in config.py


