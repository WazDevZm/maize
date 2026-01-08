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
    """Load the YOLO model with error handling"""
    global model
    try:
        # Try to load the trained model
        model_paths = [
            Path("best.pt"),
            Path("../maize_disease_app/best.pt"),
            Path("../../maize_disease_app/best.pt"),
            Path("yolov8n.pt")
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"Loading model from: {model_path}")
                
                # Fix PyTorch 2.6+ weights_only issue
                original_load = torch.load
                torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
                
                try:
                    model = YOLO(str(model_path))
                    torch.load = original_load  # Restore original
                    
                    # Ensure model has correct class names
                    if not hasattr(model, 'names') or len(model.names) != 4:
                        model.names = {
                            0: 'Health',
                            1: 'Grey_Leaf_Spots',
                            2: 'Leaf_Blight',
                            3: 'MSV'
                        }
                    
                    logger.info(f"✅ Model loaded successfully from: {model_path}")
                    logger.info(f"Model classes: {model.names}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    torch.load = original_load  # Restore original
                    logger.warning(f"Failed to load model from {model_path}: {e}")
                    continue
        
        if not model_loaded:
            logger.error("❌ No model could be loaded")
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

def predict_disease(img_array: np.ndarray, confidence_threshold: float = 0.25) -> Dict[str, Any]:
    """Run disease prediction on the image"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run inference
        results = model(img_array, conf=confidence_threshold, verbose=False)
        result = results[0]
        
        # Extract predictions
        predictions = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            for class_id, conf in zip(class_ids, confidences):
                class_name = model.names[class_id]
                predictions.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'disease_info': DISEASE_INFO.get(class_name, {}),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Create response
        response = {
            'success': True,
            'predictions_count': len(predictions),
            'detailed_predictions': predictions,
            'health_status': 'Healthy' if not predictions or (len(predictions) == 1 and predictions[0]['class'] == 'Health') else 'Disease Detected',
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': 'YOLOv8',
                'classes': model.names,
                'confidence_threshold': confidence_threshold
            }
        }
        
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
    confidence: float = Form(0.25)
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
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Run prediction
        result = predict_disease(img_array, confidence)
        
        logger.info(f"Detection completed for {file.filename}")
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
    
    return {
        "model_type": "YOLOv8",
        "classes": model.names,
        "total_classes": len(model.names),
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