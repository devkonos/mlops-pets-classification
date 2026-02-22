"""
Monitoring and logging utilities for MLOps pipeline
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys
from pythonjsonlogger import jsonlogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.config import LOGGING_CONFIG, LOGS_PATH


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record, record, message_dict):
        super(JSONFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name


def setup_logging(name: str, log_file: str = None):
    """Setup logger with both console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG['level'])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOGGING_CONFIG['level'])
    console_formatter = logging.Formatter(LOGGING_CONFIG['format'])
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with JSON formatting
    if log_file is None:
        log_file = LOGGING_CONFIG['log_file']
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOGGING_CONFIG['level'])
    json_formatter = JSONFormatter('%(message)s')
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Request logging utilities
class RequestLogger:
    """Log API requests and responses"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            log_file = str(LOGS_PATH / 'requests.log')
        self.logger = setup_logging('api_requests', log_file)
    
    def log_request(self, method: str, endpoint: str, params: Dict = None):
        """Log incoming request"""
        self.logger.info(
            f"Request received",
            extra={
                'method': method,
                'endpoint': endpoint,
                'params': params,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_response(self, method: str, endpoint: str, status_code: int,
                    processing_time_ms: float, prediction: str = None):
        """Log response"""
        self.logger.info(
            f"Response sent",
            extra={
                'method': method,
                'endpoint': endpoint,
                'status_code': status_code,
                'processing_time_ms': processing_time_ms,
                'prediction': prediction,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_error(self, method: str, endpoint: str, error: str, 
                 status_code: int = 500):
        """Log error"""
        self.logger.error(
            f"Request error",
            extra={
                'method': method,
                'endpoint': endpoint,
                'status_code': status_code,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            }
        )


class MetricsCollector:
    """Collect and store performance metrics"""
    
    def __init__(self, metrics_file: str = None):
        if metrics_file is None:
            metrics_file = str(LOGS_PATH / 'metrics.jsonl')
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(self, prediction: str, confidence: float,
                      processing_time_ms: float, model: str = 'unknown'):
        """Log prediction metrics"""
        metric = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'prediction',
            'prediction': prediction,
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'model': model
        }
        self._write_metric(metric)
    
    def log_training(self, epoch: int, train_loss: float, train_acc: float,
                    val_loss: float, val_acc: float, model: str = 'unknown'):
        """Log training metrics"""
        metric = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'training',
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'model': model
        }
        self._write_metric(metric)
    
    def log_deployment(self, status: str, version: str, environment: str):
        """Log deployment event"""
        metric = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'deployment',
            'status': status,
            'version': version,
            'environment': environment
        }
        self._write_metric(metric)
    
    def _write_metric(self, metric: Dict):
        """Write metric to file"""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            logging.error(f"Failed to write metric: {e}")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of collected metrics"""
        if not self.metrics_file.exists():
            return {}
        
        metrics_by_type = {}
        try:
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    metric = json.loads(line)
                    metric_type = metric.get('type', 'unknown')
                    if metric_type not in metrics_by_type:
                        metrics_by_type[metric_type] = []
                    metrics_by_type[metric_type].append(metric)
        except Exception as e:
            logging.error(f"Failed to read metrics: {e}")
        
        return metrics_by_type


class PerformanceAnalyzer:
    """Analyze model performance metrics from logs"""
    
    @staticmethod
    def get_prediction_stats(metrics_file: str) -> Dict:
        """Get statistics from prediction logs"""
        predictions = {'cats': 0, 'dogs': 0}
        processing_times = []
        confidences = []
        
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    try:
                        metric = json.loads(line)
                        if metric.get('type') == 'prediction':
                            predictions[metric.get('prediction', 'unknown')] += 1
                            processing_times.append(metric.get('processing_time_ms', 0))
                            confidences.append(metric.get('confidence', 0))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return {}
        
        if not processing_times:
            return {}
        
        import numpy as np
        return {
            'total_predictions': sum(predictions.values()),
            'predictions_by_class': predictions,
            'avg_processing_time_ms': float(np.mean(processing_times)),
            'min_processing_time_ms': float(np.min(processing_times)),
            'max_processing_time_ms': float(np.max(processing_times)),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }


# Initialize global loggers
request_logger = RequestLogger()
metrics_collector = MetricsCollector()
