"""
Configuration settings for Cats vs Dogs MLOps pipeline
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_ROOT_PATH = PROJECT_ROOT / "data" / "PetImages"  # Main training data
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS_PATH = PROJECT_ROOT / "data" / "splits"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"
ARTIFACTS_PATH = MODELS_PATH / "artifacts"

# Create directories if they don't exist
for path in [DATA_ROOT_PATH, DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_SPLITS_PATH, 
             MODELS_PATH, LOGS_PATH, ARTIFACTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'num_classes': 2,
    'input_size': 224,
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'early_stopping_patience': 3,
    'test_size': 0.1,
    'val_size': 0.1,
}

# Data configuration
DATA_CONFIG = {
    'target_size': (224, 224),
    'random_state': 42,
    'test_size': 0.1,
    'val_size': 0.1,
}

# API configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 8000)),
    'debug': os.getenv('API_DEBUG', 'False').lower() == 'true',
    'reload': os.getenv('API_RELOAD', 'False').lower() == 'true',
    'workers': int(os.getenv('API_WORKERS', 4)),
}

# MLflow configuration
# Use file-based tracking by default (works in CI/CD and local without server)
MLFLOW_CONFIG = {
    'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', f'file:{PROJECT_ROOT / "mlruns"}'),
    'experiment_name': os.getenv('MLFLOW_EXPERIMENT', 'cats_vs_dogs'),
    'backend_store_uri': os.getenv('MLFLOW_BACKEND_STORE_URI', f'file:{PROJECT_ROOT / "mlruns"}'),
    'default_artifact_root': os.getenv('MLFLOW_ARTIFACT_ROOT', str(ARTIFACTS_PATH)),
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_PATH / 'app.log'),
}

# Class mapping
CLASS_NAMES = ['cats', 'dogs']
CLASS_MAPPING = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
INVERSE_CLASS_MAPPING = {idx: cls_name for cls_name, idx in CLASS_MAPPING.items()}

# Training configuration
TRAINING_CONFIG = {
    'model_type': 'resnet18',  # or 'simple_cnn', 'resnet50'
    'pretrained': True,
    'freeze_backbone': False,
}

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = ENVIRONMENT == 'development'
