"""
Unit tests for data preprocessing and model inference
"""
import pytest
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.image_dataset import preprocess_dataset, get_dataloaders
from src.models.train import SimpleConvNet
from src.config import CLASS_NAMES, MODEL_CONFIG


class TestDataPreprocessing:
    """Tests for data preprocessing functions"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test images"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for cls in ['cats', 'dogs']:
                cls_dir = Path(temp_dir) / split / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                for i in range(2):
                    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
                    img.save(cls_dir / f"test_{i}.jpg")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_preprocess_dataset_creates_correct_structure(self, temp_data_dir):
        """Test that preprocessing maintains directory structure"""
        result = preprocess_dataset(temp_data_dir)
        assert result == True
        
        # Check images exist
        for split in ['train', 'val', 'test']:
            for cls in ['cats', 'dogs']:
                cls_dir = Path(temp_data_dir) / split / cls
                images = list(cls_dir.glob('*.jpg'))
                assert len(images) > 0, f"No images in {cls_dir}"
    
    def test_preprocess_handles_corrupt_images(self, temp_data_dir):
        """Test that preprocessing handles corrupt images gracefully"""
        # Create corrupt image file
        corrupt_dir = Path(temp_data_dir) / 'train' / 'cats'
        corrupt_path = corrupt_dir / 'corrupt.jpg'
        corrupt_path.write_bytes(b'This is not an image')
        
        # Should handle without crashing
        result = preprocess_dataset(temp_data_dir)
        assert result == True
        assert not corrupt_path.exists(), "Corrupt image should be removed"
    
    def test_preprocess_resize_to_224x224(self, temp_data_dir):
        """Test that images are resized correctly"""
        # Create image with different size
        test_img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        img_path = Path(temp_data_dir) / 'train' / 'cats' / 'test_original.jpg'
        test_img.save(img_path)
        
        preprocess_dataset(temp_data_dir)
        
        # Check resized image
        processed_img = Image.open(img_path)
        assert processed_img.size == (224, 224), f"Image size is {processed_img.size}, not 224x224"


class TestDataLoaders:
    """Tests for DataLoader creation"""
    
    @pytest.fixture
    def temp_data_dir_with_images(self):
        """Create temporary directory with structured images"""
        temp_dir = tempfile.mkdtemp()
        
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            for cls, label in [('cats', 0), ('dogs', 1)]:
                cls_dir = Path(temp_dir) / split / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                
                # Create 5 dummy images per class
                for i in range(5):
                    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
                    img.save(cls_dir / f"image_{i}.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_get_dataloaders_returns_dict(self, temp_data_dir_with_images):
        """Test that get_dataloaders returns correct structure"""
        dataloaders = get_dataloaders(temp_data_dir_with_images, batch_size=2)
        
        assert isinstance(dataloaders, dict)
        assert 'train' in dataloaders
        assert 'val' in dataloaders
        assert 'test' in dataloaders
    
    def test_dataloader_batch_dimensions(self, temp_data_dir_with_images):
        """Test that batches have correct dimensions"""
        dataloaders = get_dataloaders(temp_data_dir_with_images, batch_size=2)
        
        # Get first batch from train loader
        for batch_images, batch_labels in dataloaders['train']:
            assert batch_images.shape[0] <= 2  # batch size
            assert batch_images.shape[1] == 3  # RGB channels
            assert batch_images.shape[2] == 224  # height
            assert batch_images.shape[3] == 224  # width
            assert len(batch_labels) == batch_images.shape[0]
            break
    
    def test_dataloader_has_correct_labels(self, temp_data_dir_with_images):
        """Test that labels are 0 (cats) or 1 (dogs)"""
        dataloaders = get_dataloaders(temp_data_dir_with_images, batch_size=2)
        
        for batch_images, batch_labels in dataloaders['train']:
            assert torch.all((batch_labels == 0) | (batch_labels == 1))
            break


class TestModelInference:
    """Tests for model inference"""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing"""
        model = SimpleConvNet(num_classes=2)
        model.eval()
        return model
    
    def test_model_forward_pass(self, simple_model):
        """Test that model forward pass works"""
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = simple_model(x)
        
        assert output.shape == (batch_size, 2), f"Output shape is {output.shape}"
    
    def test_model_output_shape(self, simple_model):
        """Test model output has correct shape"""
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = simple_model(x)
        
        assert output.shape == (1, 2)
    
    def test_model_output_softmax(self, simple_model):
        """Test that softmax output is valid probability"""
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            logits = simple_model(x)
            probs = torch.softmax(logits, dim=1)
        
        # Check probabilities sum to 1
        sums = torch.sum(probs, dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        
        # Check all probs are between 0 and 1
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_model_prediction_consistency(self, simple_model):
        """Test that same input gives same output"""
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            out1 = simple_model(x)
            out2 = simple_model(x)
        
        assert torch.allclose(out1, out2)


class TestImageFileHandling:
    """Tests for image file handling"""
    
    def test_create_dummy_image(self):
        """Test creating a dummy image"""
        img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        assert img.size == (224, 224)
        assert img.mode == 'RGB'
    
    def test_image_format_conversion(self):
        """Test image format conversion"""
        img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        
        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Load from bytes
        loaded_img = Image.open(buffer)
        assert loaded_img.mode == 'RGB'
        assert loaded_img.size == (224, 224)


class TestConfigAndConstants:
    """Tests for configuration validation"""
    
    def test_class_names_correct(self):
        """Test that class names are correctly defined"""
        assert len(CLASS_NAMES) == 2
        assert 'Cat' in CLASS_NAMES
        assert 'Dog' in CLASS_NAMES
    
    def test_model_config_has_required_keys(self):
        """Test that model config has all required keys"""
        required_keys = [
            'random_state', 'num_classes', 'input_size',
            'batch_size', 'num_epochs', 'learning_rate'
        ]
        for key in required_keys:
            assert key in MODEL_CONFIG, f"Missing config key: {key}"
        
        assert MODEL_CONFIG['num_classes'] == 2
        assert MODEL_CONFIG['input_size'] == 224


# Integration tests
class TestIntegration:
    """Integration tests"""
    
    @pytest.fixture
    def temp_data_dir_full(self):
        """Create full test dataset"""
        temp_dir = tempfile.mkdtemp()
        
        for split in ['train', 'val', 'test']:
            for cls in ['cats', 'dogs']:
                cls_dir = Path(temp_dir) / split / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                
                # Create 3 images per class per split
                for i in range(3):
                    img = Image.new('RGB', (300, 300), color=(100, 100, 100))
                    img.save(cls_dir / f"img_{i}.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_preprocessing_to_inference_pipeline(self, temp_data_dir_full):
        """Test full pipeline from preprocessing to inference"""
        # Preprocess
        preprocess_dataset(temp_data_dir_full)
        
        # Load data
        dataloaders = get_dataloaders(temp_data_dir_full, batch_size=2)
        
        # Create model
        model = SimpleConvNet(num_classes=2)
        model.eval()
        
        # Run inference on batch
        with torch.no_grad():
            for images, labels in dataloaders['test']:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
                # Verify predictions
                assert len(predictions) == len(labels)
                assert torch.all(predictions >= 0) and torch.all(predictions < 2)
                break
