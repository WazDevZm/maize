"""
FastAPI Backend Server for Maize Disease Detection
Provides REST API endpoints for the mobile app
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from ultralytics import YOLO
import base64
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Maize Disease Detection API",
    description="AI-powered maize disease detection service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease information
DISEASE_INFO = {
    "Health": {
        "description": "Healthy maize leaf with no visible disease symptoms",
        "symptoms": ["Green color", "No spots", "Normal texture", "No lesions"],
        "treatment": "Continue current care practices",
        "severity": "None",
        "color": "#28a745"
    },
    "Grey_Leaf_Spots": {
        "description": "Grey leaf spot disease caused by Cercospora zeae-maydis",
        "symptoms": ["Grey spots", "Lesions on leaves", "Yellowing", "Reduced photosynthesis"],
        "treatment": "Apply fungicides, improve air circulation, remove infected leaves",
        "severity": "Medium",
        "color": "#ffc107"
    },
    "Leaf_Blight": {
        "description": "Leaf blight disease caused by Helminthosporium maydis",
        "symptoms": ["Brown lesions", "Leaf wilting", "Yellow halos", "Premature leaf death"],
        "treatment": "Apply copper-based fungicides, improve drainage, crop rotation",
        "severity": "High",
        "color": "#dc3545"
    },
    "MSV": {
        "description": "Maize Streak Virus transmitted by leafhoppers",
        "symptoms": ["Yellow streaks", "Stunted growth", "Mosaic patterns", "Reduced yield"],
        "treatment": "Control leafhoppers, use resistant varieties, remove infected plants",
        "severity": "High",
        "color": "#dc3545"
    }
}

# Global model variable
model = None

def load_model():
    """Load the YOLO model with comprehensive error handling"""
    global model
    try:
        # Prioritize the trained best.pt model
        model_paths = [
            Path("best.pt"),
            Path("../maize_disease_app/best.pt"),
            Path("../../maize_disease_app/best.pt"),
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"Loading trained model from: {model_path}")
                
                try:
                    # Strategy 1: Try with DFLoss compatibility fix
                    try:
                        from ultralytics.utils.loss import DFLoss
                    except (ImportError, AttributeError):
                        # Create a mock DFLoss class
                        class DFLoss:
                            def __init__(self, *args, **kwargs):
                                pass
                            def __call__(self, *args, **kwargs):
                                return torch.tensor(0.0)
                        
                        # Add it to the ultralytics.utils.loss module
                        import ultralytics.utils.loss
                        ultralytics.utils.loss.DFLoss = DFLoss
                        logger.info("📝 Added compatibility layer for DFLoss")
                    
                    # Fix PyTorch 2.6+ weights_only issue
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
                    
                    try:
                        model = YOLO(str(model_path))
                        torch.load = original_load  # Restore original
                        
                        # Verify this is our trained model by checking class names
                        if hasattr(model, 'names') and len(model.names) == 4:
                            logger.info(f"✅ Trained model loaded successfully from: {model_path}")
                            logger.info(f"Model classes: {model.names}")
                            model_loaded = True
                            break
                        else:
                            # Set correct class names for our custom model
                            model.model.names = {
                                0: 'Health',
                                1: 'Grey_Leaf_Spots',
                                2: 'Leaf_Blight',
                                3: 'MSV'
                            }
                            logger.info(f"✅ Trained model loaded and configured from: {model_path}")
                            logger.info(f"Model classes: {model.model.names}")
                            model_loaded = True
                            break
                        
                    except Exception as load_error:
                        torch.load = original_load  # Restore original
                        raise load_error
                    
                except Exception as e:
                    logger.warning(f"Failed to load trained model from {model_path}: {e}")
                    continue
        
        # Only fall back to base model if trained model fails
        if not model_loaded:
            logger.warning("Trained model not available, falling back to base YOLOv8...")
            try:
                base_model_path = Path("yolov8n.pt")
                if base_model_path.exists():
                    model = YOLO(str(base_model_path))
                    # Configure for our classes (this won't have trained weights)
                    model.model.names = {
                        0: 'Health',
                        1: 'Grey_Leaf_Spots',
                        2: 'Leaf_Blight',
                        3: 'MSV'
                    }
                    logger.warning("⚠️ Using base YOLOv8n model - predictions may be inaccurate")
                    model_loaded = True
                else:
                    raise FileNotFoundError("No model files found")
            except Exception as e:
                logger.error(f"Base model loading failed: {e}")
                return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical error loading model: {e}")
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for model inference"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV/YOLO
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

def predict_disease(img_array: np.ndarray, confidence_threshold: float = 0.15) -> Dict[str, Any]:
    """Run disease prediction on the image with optimized confidence"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run inference with lower confidence threshold for better sensitivity
        results = model(img_array, conf=confidence_threshold, verbose=False)
        result = results[0]
        
        # Extract predictions
        predictions = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            # Get model names - handle both real and mock models
            model_names = getattr(model, 'names', None)
            if model_names is None and hasattr(model, 'model'):
                model_names = getattr(model.model, 'names', None)
            
            if model_names is None:
                # Fallback names
                model_names = {
                    0: 'Health',
                    1: 'Grey_Leaf_Spots',
                    2: 'Leaf_Blight',
                    3: 'MSV'
                }
            
            for class_id, conf in zip(class_ids, confidences):
                class_name = model_names[class_id]
                predictions.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'disease_info': DISEASE_INFO.get(class_name, {}),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Sort predictions by confidence (highest first)
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Determine health status based on predictions
        if not predictions:
            health_status = 'Healthy'
            severity = 'None'
        elif len(predictions) == 1 and predictions[0]['class'] == 'Health':
            health_status = 'Healthy'
            severity = 'None'
        else:
            # Find the highest confidence non-health prediction
            disease_predictions = [p for p in predictions if p['class'] != 'Health']
            if disease_predictions:
                best_disease = disease_predictions[0]
                health_status = f'Disease detected: {best_disease["class"]}'
                severity = DISEASE_INFO.get(best_disease['class'], {}).get('severity', 'Unknown')
            else:
                health_status = 'Healthy'
                severity = 'None'
        
        # Create response
        response = {
            'success': True,
            'predictions_count': len(predictions),
            'detailed_predictions': predictions,
            'health_status': health_status,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': 'YOLOv8',
                'classes': model_names if 'model_names' in locals() else {},
                'confidence_threshold': confidence_threshold,
                'model_path': 'best.pt' if hasattr(model, 'model') else 'unknown'
            }
        }
        
        logger.info(f"Prediction completed: {len(predictions)} detections, health_status: {health_status}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("🚀 Starting Maize Disease Detection API...")
    success = load_model()
    if success:
        logger.info("✅ API ready to serve requests")
    else:
        logger.error("❌ API started but model loading failed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Maize Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

@app.post("/detect")
async def detect_disease(
    file: UploadFile = File(...),
    confidence: float = Form(0.15)  # Lower default confidence for better sensitivity
):
    """
    Detect diseases in a single maize leaf image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Log image info
        logger.info(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes, confidence: {confidence}")
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Run prediction
        result = predict_disease(img_array, confidence)
        
        logger.info(f"Detection completed for {file.filename}: {result['predictions_count']} predictions")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/detect-batch")
async def detect_diseases_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Form(0.25)
):
    """
    Detect diseases in multiple maize leaf images
    """
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
        
        batch_results = []
        
        for i, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    batch_results.append({
                        'image_index': i,
                        'filename': file.filename,
                        'success': False,
                        'error': 'Invalid file type'
                    })
                    continue
                
                # Read and process image
                image_bytes = await file.read()
                img_array = preprocess_image(image_bytes)
                
                # Run prediction
                result = predict_disease(img_array, confidence)
                result['image_index'] = i
                result['filename'] = file.filename
                
                batch_results.append(result)
                
            except Exception as e:
                batch_results.append({
                    'image_index': i,
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return JSONResponse(content={
            'success': True,
            'batch_results': batch_results,
            'total_images': len(files),
            'timestamp': datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

@app.get("/diseases")
async def get_diseases():
    """Get information about all diseases"""
    return {
        "diseases": DISEASE_INFO,
        "total_diseases": len(DISEASE_INFO),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/diseases/{disease_name}")
async def get_disease_info(disease_name: str):
    """Get information about a specific disease"""
    if disease_name not in DISEASE_INFO:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    return {
        "disease": disease_name,
        "info": DISEASE_INFO[disease_name],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get model names - handle both real and mock models
    model_names = getattr(model, 'names', None)
    if model_names is None and hasattr(model, 'model'):
        model_names = getattr(model.model, 'names', None)
    
    if model_names is None:
        # Fallback names
        model_names = {
            0: 'Health',
            1: 'Grey_Leaf_Spots',
            2: 'Leaf_Blight',
            3: 'MSV'
        }
    
    return {
        "model_type": "YOLOv8",
        "classes": model_names,
        "total_classes": len(model_names),
        "status": "loaded",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )