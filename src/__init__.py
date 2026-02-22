"""
Cats vs Dogs MLOps Pipeline
End-to-end ML pipeline for binary image classification
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"

from src.config import (
    MODEL_CONFIG,
    API_CONFIG,
    MLFLOW_CONFIG,
    CLASS_NAMES,
    CLASS_MAPPING,
)

__all__ = [
    'MODEL_CONFIG',
    'API_CONFIG',
    'MLFLOW_CONFIG',
    'CLASS_NAMES',
    'CLASS_MAPPING',
]
