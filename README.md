# MNIST CNN Classification Model

A PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification.

## Model Architecture

The model uses a modern CNN architecture with the following structure:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
       BatchNorm2d-2           [-1, 16, 26, 26]              32
              ReLU-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 24, 24, 24]           3,456
       BatchNorm2d-5           [-1, 24, 24, 24]              48
              ReLU-6           [-1, 24, 24, 24]               0
            Conv2d-7           [-1, 24, 24, 24]           5,184
       BatchNorm2d-8           [-1, 24, 24, 24]              48
              ReLU-9           [-1, 24, 24, 24]               0
           Conv2d-10           [-1, 16, 24, 24]           3,456
      BatchNorm2d-11           [-1, 16, 24, 24]              32
             ReLU-12           [-1, 16, 24, 24]               0
        MaxPool2d-13           [-1, 16, 12, 12]               0
           Conv2d-14           [-1, 16, 12, 12]           2,304
      BatchNorm2d-15           [-1, 16, 12, 12]              32
             ReLU-16           [-1, 16, 12, 12]               0
           Conv2d-17           [-1, 24, 12, 12]           3,456
      BatchNorm2d-18           [-1, 24, 12, 12]              48
             ReLU-19           [-1, 24, 12, 12]               0
           Conv2d-20           [-1, 16, 10, 10]           3,456
      BatchNorm2d-21           [-1, 16, 10, 10]              32
             ReLU-22           [-1, 16, 10, 10]               0
           Conv2d-23             [-1, 10, 8, 8]           1,440
      BatchNorm2d-24             [-1, 10, 8, 8]              20
             ReLU-25             [-1, 10, 8, 8]               0
        AvgPool2d-26             [-1, 10, 4, 4]               0
           Conv2d-27             [-1, 10, 1, 1]           1,610
================================================================
Total params: 24,798
Trainable params: 24,798
Non-trainable params: 0
----------------------------------------------------------------
```

### Key Features:
- Initial feature extraction (1→16 channels)
- Deep feature processing (16→24→24→16 channels)
- Spatial reduction using MaxPool2d
- Feature refinement with multiple conv blocks
- Global Average Pooling for classification
- Final 1x1 convolution to output classes

## Training Configuration

- **Dataset**: MNIST
  - Training: 60,000 images
  - Testing: 10,000 images
  - Mean: 0.1307
  - Std: 0.3015

- **Data Augmentation**:
  - Random Rotation (±5°)
  - Normalization (mean=0.1307, std=0.3015)

- **Hyperparameters**:
  - Optimizer: Adam
    - Initial LR: 0.001
    - Betas: (0.9, 0.999)
    - Weight Decay: 1e-5
  - Scheduler: OneCycleLR
    - Max LR: 0.01
    - Pct Start: 0.2
    - Div Factor: 10.0
    - Final Div Factor: 100.0
    - Anneal Strategy: 'cos'
  - Batch Size: 128
  - Workers: 4
  - Pin Memory: True

## Model Validation

The model meets the following criteria:
- Parameters: 24,798 (< 25,000 requirement)
- Training Accuracy after 1 epoch: 95.06%
- Uses BatchNorm and Global Average Pooling

## Requirements

```
torch==1.13.0
torchvision==0.14.0
numpy<2
tqdm==4.64.0
matplotlib==3.5.1
```

## Project Structure
```
assignment7/
├── src/
│   ├── model.py      # Model architecture
│   └── train.py      # Training pipeline
├── requirements.txt  # Dependencies
├── .gitignore        # Git ignore file
├── .github/
│   └── workflows/
│       └── ci-cd.yml # GitHub Actions workflow
└── README.md         # Project documentation
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Validates the model architecture
2. Runs training validation
3. Saves model artifacts with timestamps
4. Uploads trained models as artifacts

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python src/train.py
```

3. Monitor training metrics:
```bash
tensorboard --logdir=runs/mnist_training
```

## License

This project is open-sourced under the MIT license.
