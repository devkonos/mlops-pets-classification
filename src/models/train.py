"""
PyTorch CNN Model Training with MLflow Tracking
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.config import (
    MODEL_CONFIG, MLFLOW_CONFIG, ARTIFACTS_PATH, 
    LOGGING_CONFIG, CLASS_NAMES, ENVIRONMENT
)
from src.data.image_dataset import get_dataloaders

# Configure logging
_handlers = [logging.StreamHandler()]
try:
    _handlers.append(logging.FileHandler(LOGGING_CONFIG['log_file']))
except Exception as e:
    # If we can't create file handler, just use console
    print(f"[WARN] Could not create file handler: {e}")
    
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    handlers=_handlers
)
logger = logging.getLogger(__name__)


class SimpleConvNet(nn.Module):
    """Simple CNN for Cats vs Dogs classification"""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_transfer_learning_model(model_name: str = 'resnet18', 
                                num_classes: int = 2,
                                pretrained: bool = True):
    """Load pretrained model for transfer learning"""
    if model_name == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


class ModelTrainer:
    """Handles model training and evaluation with MLflow tracking"""
    
    def __init__(self, model_name: str = 'simple_cnn', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.best_metrics = {}
        
        # Setup MLflow
        try:
            mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
            mlflow.set_experiment(MLFLOW_CONFIG['experiment_name'])
            logger.info(f"MLflow tracking URI: {MLFLOW_CONFIG['tracking_uri']}")
        except Exception as e:
            logger.warning(f"MLflow setup failed (tracking may be disabled): {e}")
    
    def build_model(self):
        """Build the model"""
        if self.model_name == 'simple_cnn':
            self.model = SimpleConvNet(num_classes=MODEL_CONFIG['num_classes'])
        else:
            self.model = get_transfer_learning_model(
                self.model_name,
                MODEL_CONFIG['num_classes'],
                pretrained=True
            )
        
        self.model = self.model.to(self.device)
        logger.info(f"Model created: {self.model_name}")
        return self.model
    
    def train_epoch(self, dataloader, criterion, optimizer):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, dataloader, criterion):
        """Evaluate model on validation/test set"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
        }
    
    def train(self, data_dir: str, num_epochs: int = None,
              batch_size: int = None, learning_rate: float = None):
        """Train model with MLflow tracking"""
        num_epochs = num_epochs or MODEL_CONFIG['num_epochs']
        batch_size = batch_size or MODEL_CONFIG['batch_size']
        learning_rate = learning_rate or MODEL_CONFIG['learning_rate']
        
        # Load data
        logger.info(f"Loading data from {data_dir}")
        # Check if this is PetImages folder that needs splitting
        data_path = Path(data_dir)
        if (data_path / 'Cat').exists() and (data_path / 'Dog').exists():
            # PetImages folder - create splits
            dataloaders = get_dataloaders(
                data_dir, 
                batch_size=batch_size,
                splits_output='data/splits'
            )
        else:
            # Already split folders
            dataloaders = get_dataloaders(data_dir, batch_size=batch_size)
        
        # Build model
        self.build_model()
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        
        # MLflow run (wrapped for resilience)
        class SafeMlflowRun:
            def __enter__(self):
                try:
                    self.run = mlflow.start_run()
                    return self.run
                except Exception as e:
                    logger.warning(f"Could not start MLflow run: {e}")
                    return None
            
            def __exit__(self, *args):
                try:
                    if self.run:
                        mlflow.end_run()
                except:
                    pass
        
        with SafeMlflowRun() as run:
            # Log parameters
            if run:
                try:
                    mlflow.log_param('model_name', self.model_name)
                    mlflow.log_param('num_epochs', num_epochs)
                    mlflow.log_param('batch_size', batch_size)
                    mlflow.log_param('learning_rate', learning_rate)
                    mlflow.log_param('device', self.device)
                    mlflow.log_param('environment', ENVIRONMENT)
                except Exception as e:
                    logger.warning(f"Could not log MLflow params: {e}")
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            for epoch in range(num_epochs):
                logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
                
                # Train
                train_loss, train_acc = self.train_epoch(
                    dataloaders['train'], criterion, optimizer
                )
                
                # Validate
                val_metrics = self.evaluate(dataloaders['val'], criterion)
                
                # Log metrics (safe)
                try:
                    if run:
                        mlflow.log_metrics({
                            'train_loss': train_loss,
                            'train_accuracy': train_acc,
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy'],
                            'val_precision': val_metrics['precision'],
                            'val_recall': val_metrics['recall'],
                            'val_f1': val_metrics['f1'],
                        }, step=epoch)
                except Exception as e:
                    logger.debug(f"Could not log MLflow metrics: {e}")
                
                logger.info(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
                )
                
                # Early stopping
                scheduler.step(val_metrics['loss'])
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self.best_metrics = val_metrics.copy()
                    
                    # Save best model
                    model_path = ARTIFACTS_PATH / f"{self.model_name}_best.pt"
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"Best model saved to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= MODEL_CONFIG['early_stopping_patience']:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Final evaluation on test set
            logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(dataloaders['test'], criterion)
            
            # Log test metrics (safe)
            try:
                if run:
                    mlflow.log_metrics({
                        'test_loss': test_metrics['loss'],
                        'test_accuracy': test_metrics['accuracy'],
                        'test_precision': test_metrics['precision'],
                        'test_recall': test_metrics['recall'],
                        'test_f1': test_metrics['f1'],
                    })
                    
                    # Log confusion matrix
                    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
                    cm_dict = {
                        'confusion_matrix': cm.tolist(),
                        'labels': CLASS_NAMES,
                    }
                    mlflow.log_dict(cm_dict, 'confusion_matrix.json')
                    
                    # Log classification report
                    class_report = classification_report(
                        test_metrics['labels'],
                        test_metrics['predictions'],
                        target_names=CLASS_NAMES,
                        output_dict=True
                    )
                    mlflow.log_dict(class_report, 'classification_report.json')
            except Exception as e:
                logger.debug(f"Could not log MLflow test metrics: {e}")
            
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
            
            # Log model (safe)
            try:
                if run:
                    mlflow.pytorch.log_model(
                        self.model, "model",
                        artifact_path=str(ARTIFACTS_PATH)
                    )
                    logger.info(f"Model trained successfully. Run ID: {mlflow.active_run().info.run_id}")
            except Exception as e:
                logger.debug(f"Could not log MLflow model: {e}")
                logger.info("Model trained successfully (MLflow logging skipped)")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs model')
    parser.add_argument('--data-dir', type=str, default='data/splits',
                      help='Path to data splits')
    parser.add_argument('--model', type=str, default='simple_cnn',
                      choices=['simple_cnn', 'resnet18', 'resnet50'],
                      help='Model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to use')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    logger.info("Starting model training...")
    trainer = ModelTrainer(model_name=args.model, device=args.device)
    trainer.train(
        args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
