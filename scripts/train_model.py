"""
Main model training script - REFERENCE pattern
"""
import os
import sys
import logging
from pathlib import Path

# Setup paths - ADD PROJECT ROOT (like REFERENCE does)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import (
    MODEL_CONFIG, MLFLOW_CONFIG, ARTIFACTS_PATH, 
    LOGGING_CONFIG, CLASS_NAMES
)
from src.models.train import ModelTrainer
from src.data.image_dataset import get_dataloaders

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    try:
        logger.info("=" * 70)
        logger.info("STARTING MODEL TRAINING PIPELINE - CATS VS DOGS")
        logger.info("=" * 70)
        
        # Get dataloaders
        logger.info(f"Loading data from data/PetImages/")
        dataloaders = get_dataloaders(
            data_dir='data/PetImages',
            batch_size=MODEL_CONFIG['batch_size'],
            splits_output='data/splits'
        )
        
        # Initialize trainer
        logger.info(f"Initializing model trainer with: {MODEL_CONFIG}")
        trainer = ModelTrainer(
            model_name='simple_cnn',
            device='cpu'  # GitHub Actions uses CPU
        )
        
        # Train model
        logger.info("Starting model training...")
        trainer.train(
            data_dir='data/PetImages',
            num_epochs=MODEL_CONFIG['num_epochs'],
            batch_size=MODEL_CONFIG['batch_size'],
            learning_rate=MODEL_CONFIG['learning_rate']
        )
        
        logger.info("=" * 70)
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
