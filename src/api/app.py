"""
FastAPI application for Cats vs Dogs model inference service
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
import sys
import time
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    API_CONFIG, ARTIFACTS_PATH, LOGGING_CONFIG,
    CLASS_NAMES, INVERSE_CLASS_MAPPING, MODEL_CONFIG
)
from src.models.train import SimpleConvNet, get_transfer_learning_model
from src.data.image_dataset import preprocess_dataset

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

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="ML Model API for binary image classification",
    version="1.0.0"
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['prediction_class']
)

REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy metric'
)

PREDICTIONS_CATS = Gauge(
    'predictions_cats_total',
    'Total cat predictions'
)

PREDICTIONS_DOGS = Gauge(
    'predictions_dogs_total',
    'Total dog predictions'
)

# Model loading
MODEL_PATH = ARTIFACTS_PATH / "simple_cnn_best.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
model_loaded = False

def load_model():
    """Load trained model"""
    global model, model_loaded
    try:
        model = SimpleConvNet(num_classes=len(CLASS_NAMES))
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            model_loaded = True
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            model_loaded = False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting FastAPI application...")
    try:
        load_model()
        if model_loaded:
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model not loaded - will run in degraded mode")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Don't crash the app - continue with degraded mode
        pass


# Request/Response models
class PredictionInput(BaseModel):
    """Input model for file upload"""
    pass


class PredictionOutput(BaseModel):
    """Output model for prediction response"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Cats vs Dogs Classification API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status=200).inc()
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=str(DEVICE),
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict", tags=["Prediction"], response_model=PredictionOutput)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image.
    
    Args:
        file: Image file (JPEG/PNG)
    
    Returns:
        Prediction with confidence and probabilities
    """
    start_time = time.time()
    
    try:
        if not model_loaded:
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=503).inc()
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server health."
            )
        
        # Verify file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=400).inc()
            raise HTTPException(
                status_code=400,
                detail="File must be JPEG or PNG image"
            )
        
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Preprocess
        from torchvision import transforms
        preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = preprocessor(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
        
        # Prepare response
        prob_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Update metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=200).inc()
        PREDICTION_COUNT.labels(prediction_class=predicted_class).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(processing_time / 1000)
        
        if predicted_class == "cats":
            PREDICTIONS_CATS.inc()
        else:
            PREDICTIONS_DOGS.inc()
        
        logger.info(
            f"Prediction: {predicted_class} (confidence: {confidence:.4f}), "
            f"Processing time: {processing_time:.2f}ms"
        )
        
        return PredictionOutput(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple images.
    
    Args:
        files: List of image files
    
    Returns:
        List of predictions
    """
    if not model_loaded:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict-batch", status=503).inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    results = []
    for file in files:
        try:
            # Create a mock request for single prediction
            # In production, you'd want to optimize this
            image = Image.open(BytesIO(await file.read())).convert('RGB')
            
            from torchvision import transforms
            preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_tensor = preprocessor(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = float(probabilities[predicted_class_idx])
            
            results.append({
                'filename': file.filename,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    CLASS_NAMES[i]: float(probabilities[i])
                    for i in range(len(CLASS_NAMES))
                }
            })
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    REQUEST_COUNT.labels(method="POST", endpoint="/predict-batch", status=200).inc()
    return {'predictions': results, 'count': len(results)}


@app.get("/info", tags=["Info"])
async def info():
    """Get API and model information"""
    return {
        'api_version': '1.0.0',
        'model_loaded': model_loaded,
        'model_path': str(MODEL_PATH),
        'classes': CLASS_NAMES,
        'device': str(DEVICE),
        'timestamp': datetime.utcnow().isoformat()
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        workers=API_CONFIG['workers']
    )
