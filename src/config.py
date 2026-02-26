"""
Configuration file for Fake Image Detection System
Contains all hyperparameters and paths
"""

import os
import torch

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, 
                  MODEL_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# DATASET PARAMETERS
# ============================================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Image parameters
IMG_SIZE = 224
IMG_CHANNELS = 3

# Data loading
NUM_WORKERS = 2  # Reduce to 0 if facing issues on Windows
PIN_MEMORY = True

# ============================================
# MODEL PARAMETERS
# ============================================
MODEL_NAME = 'efficientnet_b0'
NUM_CLASSES = 1  # Binary classification (sigmoid output)
DROPOUT_RATE = 0.5

# ============================================
# TRAINING PARAMETERS
# ============================================
# Phase 1: Train only classifier
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-4
PHASE1_FREEZE_BASE = True

# Phase 2: Fine-tune entire model
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-5
PHASE2_UNFREEZE_LAYERS = 30  # Number of layers to unfreeze from the end

# Batch sizes
BATCH_SIZE = 16  # Reduce to 16 or 8 if CUDA out of memory

# Optimizer
OPTIMIZER = 'adam'
WEIGHT_DECAY = 1e-4

# Learning rate scheduler
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 5

# ============================================
# DEVICE CONFIGURATION
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()

# ============================================
# DATA AUGMENTATION
# ============================================
# Training augmentations
TRAIN_AUGMENTATION = {
    'horizontal_flip': 0.5,
    'rotation': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
}

# ImageNet normalization (required for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# EVALUATION PARAMETERS
# ============================================
CLASSIFICATION_THRESHOLD = 0.5  # For binary classification
SAVE_CONFUSION_MATRIX = True
SAVE_ROC_CURVE = True
SAVE_TRAINING_HISTORY = True

# ============================================
# PREDICTION PARAMETERS
# ============================================
CONFIDENCE_THRESHOLD = 0.8  # High confidence threshold for predictions

# ============================================
# LOGGING
# ============================================
LOG_FILE = os.path.join(LOG_DIR, 'training.log')
LOG_LEVEL = 'INFO'
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard')

# ============================================
# MODEL SAVING
# ============================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoint.pth')
SAVE_MODEL_EVERY_N_EPOCHS = 5

# ============================================
# DISPLAY SETTINGS
# ============================================
VERBOSE = True
SHOW_PLOTS = True

# ============================================
# STREAMLIT APP SETTINGS
# ============================================
APP_TITLE = "🔍 AI-Generated Image Detector"
APP_DESCRIPTION = "Upload an image to detect if it's real or AI-generated"
MAX_UPLOAD_SIZE = 10  # MB

# ============================================
# CLASS NAMES
# ============================================
CLASS_NAMES = ['Real', 'Fake']

# ============================================
# PRINT CONFIGURATION
# ============================================
def print_config():
    """Print all configuration settings"""
    print("="*60)
    print("CONFIGURATION SETTINGS")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Phase 1 - Epochs: {PHASE1_EPOCHS}, LR: {PHASE1_LR}")
    print(f"Phase 2 - Epochs: {PHASE2_EPOCHS}, LR: {PHASE2_LR}")
    print(f"Train/Val/Test Split: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    print(f"Best Model Path: {BEST_MODEL_PATH}")
    print("="*60)

if __name__ == "__main__":
    print_config()
