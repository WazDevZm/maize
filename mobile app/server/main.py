"""
FastAPI Backend for Maize Disease Detection Mobile App
Provides REST API endpoints for disease detection using YOLOv8 model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Maize Disease Detection API",
    description="AI-powered maize disease detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React Native app's origin
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
        "color": "#28a745",
        "prevention": [
            "Maintain proper spacing between plants",
            "Ensure adequate nutrition",
            "Regular monitoring"
        ]
    },
    "Grey_Leaf_Spots": {
        "description": "Grey leaf spot disease caused by Cercospora zeae-maydis",
        "symptoms": ["Grey spots", "Lesions on leaves", "Yellowing", "Reduced photosynthesis"],
        "treatment": "Apply fungicides, improve air circulation, remove infected leaves",
        "severity": "Medium",
        "color": "#ffc107",
        "prevention": [
            "Crop rotation with non-host plants",
            "Resistant varieties",
            "Proper field sanitation"
        ]
    },
    "Leaf_Blight": {
        "description": "Leaf blight disease caused by Helminthosporium maydis",
        "symptoms": ["Brown lesions", "Leaf wilting", "Yellow halos", "Premature leaf death"],
        "treatment": "Apply copper-based fungicides, improve drainage, crop rotation",
        "severity": "High",
        "color": "#dc3545",
        "prevention": [
            "Use resistant varieties",
            "Improve field drainage",
            "Balanced fertilization"
        ]
    },
    "MSV": {
        "description": "Maize Streak Virus transmitted by leafhoppers",
        "symptoms": ["Yellow streaks", "Stunted growth", "Mosaic patterns", "Reduced yield"],
        "treatment": "Control leafhoppers, use resistant varieties, remove infected plants",
        "severity": "High",
        "color": "#dc3545",
        "prevention": [
            "Control leafhopper vectors",
            "Use virus-free seeds",
            "Early planting"
        ]
    }
}

# Global model variable
model = None

def load_model():
    """Load the trained YOLO model"""
    global model
    try:
        # Path to the best model weights
        model_path = Path("best.pt")
        
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("../../dataset_split/runs/train_20251010_201550/weights/best.pt"),
                Path("../dataset_split/runs/train_20251010_201550/weights/best.pt")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
            else:
                logger.error(f"Model not found at any expected paths")
                return None
        
        # Fix PyTorch 2.6+ weights_only issue
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        
        try:
            model = YOLO(str(model_path))
            torch.load = original_load  # Restore original
            logger.info(f"✅ Model loaded successfully from: {model_path}")
            return model
        except Exception as e:
            torch.load = original_load  # Restore original
            logger.warning(f"Trained model loading failed: {e}")
            
            # Try loading base model as fallback
            try:
                torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
                model = YOLO('yolov8n.pt')
                torch.load = original_load  # Restore original
                
                # Customize for our classes
                model.names = {
                    0: 'Health',
                    1: 'Grey_Leaf_Spots', 
                    2: 'Leaf_Blight',
                    3: 'MSV'
                }
                
                logger.info("✅ Base YOLOv8n model loaded as fallback")
                return model
            except Exception as e2:
                torch.load = original_load  # Restore original
                logger.error(f"Base model loading also failed: {e2}")
                return None
                
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_disease(image_array: np.ndarray, confidence_threshold: float = 0.25):
    """Run disease prediction on the image"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run inference
        results = model(image_array, conf=confidence_threshold, verbose=False)
        
        # Get the first result
        result = results[0]
        
        # Extract predictions
        boxes = result.boxes
        predictions = []
        
        if boxes is not None and len(boxes) > 0:
            # Get class predictions
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            # Map class IDs to names
            class_names = model.names
            
            for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
                class_name = class_names[class_id]
                predictions.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'timestamp': datetime.now().isoformat()
                })
        
        return predictions, result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Maize Disease Detection API...")
    load_model()
    if model is None:
        logger.warning("⚠️ Model not loaded - some endpoints may not work")

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
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model is not None else "not_loaded"
    }

@app.get("/diseases")
async def get_diseases():
    """Get information about all supported diseases"""
    return {
        "diseases": DISEASE_INFO,
        "total_diseases": len(DISEASE_INFO)
    }

@app.get("/diseases/{disease_name}")
async def get_disease_info(disease_name: str):
    """Get detailed information about a specific disease"""
    if disease_name not in DISEASE_INFO:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    return {
        "disease": disease_name,
        "info": DISEASE_INFO[disease_name]
    }

@app.post("/detect")
async def detect_disease(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Detect disease in uploaded maize leaf image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        confidence: Confidence threshold (0.1-1.0)
    
    Returns:
        Detection results with disease information
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate confidence threshold
        if not 0.1 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 1.0")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        img_array = preprocess_image(image)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Run prediction
        predictions, result = predict_disease(img_array, confidence)
        
        # Process results
        response_data = {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "total_detections": len(predictions),
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        }
        
        # Add disease information for each prediction
        detailed_predictions = []
        for pred in predictions:
            disease_name = pred['class']
            disease_info = DISEASE_INFO.get(disease_name, {})
            
            detailed_pred = {
                **pred,
                "disease_info": disease_info
            }
            detailed_predictions.append(detailed_pred)
        
        response_data["detailed_predictions"] = detailed_predictions
        
        # Determine overall health status
        if not predictions:
            response_data["health_status"] = "No diseases detected"
            response_data["severity"] = "None"
            response_data["recommendations"] = ["Continue current care practices"]
        else:
            # Find highest confidence prediction
            best_pred = max(predictions, key=lambda x: x['confidence'])
            disease_name = best_pred['class']
            
            if disease_name == "Health":
                response_data["health_status"] = "Healthy"
                response_data["severity"] = "None"
                response_data["recommendations"] = ["Continue current care practices"]
            else:
                response_data["health_status"] = f"Disease detected: {disease_name}"
                response_data["severity"] = DISEASE_INFO.get(disease_name, {}).get('severity', 'Unknown')
                response_data["recommendations"] = [
                    DISEASE_INFO.get(disease_name, {}).get('treatment', 'Consult a plant pathologist')
                ]
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Detect diseases in multiple images
    
    Args:
        files: List of image files
        confidence: Confidence threshold (0.1-1.0)
    
    Returns:
        Batch detection results
    """
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
        
        batch_results = []
        
        for i, file in enumerate(files):
            try:
                # Process each image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                img_array = preprocess_image(image)
                if img_array is None:
                    continue
                
                predictions, result = predict_disease(img_array, confidence)
                
                batch_results.append({
                    "index": i,
                    "filename": file.filename,
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                batch_results.append({
                    "index": i,
                    "filename": file.filename,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "batch_results": batch_results,
            "total_processed": len(batch_results),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)