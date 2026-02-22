"""
Pytest configuration and fixtures for Cats vs Dogs MLOps project
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_data_root():
    """Create temporary root directory for all tests"""
    temp_dir = tempfile.mkdtemp(prefix="mlops_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dataset_dir(temp_data_root):
    """Create temporary directory with test dataset structure"""
    dataset_dir = Path(temp_data_root) / "dataset"
    
    # Create structure: train/val/test -> cats/dogs
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            cls_dir = dataset_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 3 dummy images per class per split
            for i in range(3):
                img = Image.new('RGB', (224, 224), color=(100, 100, 100))
                img.save(cls_dir / f"img_{i}.jpg")
    
    return dataset_dir


@pytest.fixture
def temp_image_file(temp_data_root):
    """Create a temporary image file"""
    img_dir = Path(temp_data_root) / "images"
    img_dir.mkdir(exist_ok=True)
    
    img = Image.new('RGB', (224, 224), color=(75, 150, 75))
    img_path = img_dir / "test_image.jpg"
    img.save(img_path)
    
    return img_path


@pytest.fixture
def device():
    """Get device for testing (CPU or GPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_model(device):
    """Create a simple test model"""
    from src.models.train import SimpleConvNet
    model = SimpleConvNet(num_classes=2)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def dummy_batch(device):
    """Create dummy batch of images and labels"""
    images = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 0, 1]).to(device)
    return images, labels


@pytest.fixture(autouse=True)
def log_test_name(request):
    """Log test name for debugging"""
    print(f"\n⏱️  Running test: {request.node.name}")
    yield
    print(f"✅ Completed: {request.node.name}")


# Pytest configuration options
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


# Collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests based on module
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
