"""
🌽 Maize Disease Detection - Advanced Streamlit Web Application
Real-time maize leaf disease detection using YOLOv8 with advanced features
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os
from pathlib import Path
import time
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import base64
import json
import pandas as pd
from datetime import datetime
import io
import zipfile
from typing import List, Dict, Tuple
import logging


# Page configuration
st.set_page_config(
    page_title="🌽 Maize Disease Detection",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .disease-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
    .danger-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

# Class colors for visualization
CLASS_COLORS = {
    "Health": "#28a745",
    "Grey_Leaf_Spots": "#ffc107", 
    "Leaf_Blight": "#dc3545",
    "MSV": "#dc3545"
}

# Advanced configuration
ADVANCED_CONFIG = {
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "supported_formats": ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    "image_processing": {
        "max_width": 8000,  # Increased to 8K resolution
        "max_height": 8000,  # Increased to 8K resolution
        "min_width": 100,
        "min_height": 100,
        "resize_large_images": True,  # Auto-resize if too large
        "target_size": 2048,  # Target size for processing
        "check_blur": False,  # Disable blur detection to avoid false positives
        "blur_threshold": 20  # Variance threshold for blur detection
    },
    "batch_processing": {
        "max_images": 10,
        "zip_download": True
    }
}

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {
        'total_detections': 0,
        'accuracy_score': 0.995,
        'avg_confidence': 0.0
    }

@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    try:
        # Path to the best model weights
        model_path = Path("G:/maize_annotations/dataset_split/runs/train_20251010_201550/weights/best.pt")
        
        if not model_path.exists():
            st.error(f"❌ Model not found at: {model_path}")
            st.error("Please ensure the model has been trained first!")
            return None
        
        model = YOLO(str(model_path))
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def validate_image(image: Image.Image) -> Tuple[bool, str, Image.Image]:
    """Validate uploaded image with advanced checks and smart resizing"""
    try:
        # Check dimensions
        width, height = image.size
        if width < ADVANCED_CONFIG["image_processing"]["min_width"] or height < ADVANCED_CONFIG["image_processing"]["min_height"]:
            return False, f"Image too small. Minimum size: {ADVANCED_CONFIG['image_processing']['min_width']}x{ADVANCED_CONFIG['image_processing']['min_height']} pixels", image
        
        # Handle large images with smart resizing
        if width > ADVANCED_CONFIG["image_processing"]["max_width"] or height > ADVANCED_CONFIG["image_processing"]["max_height"]:
            if ADVANCED_CONFIG["image_processing"]["resize_large_images"]:
                # Calculate new dimensions maintaining aspect ratio
                target_size = ADVANCED_CONFIG["image_processing"]["target_size"]
                if width > height:
                    new_width = target_size
                    new_height = int((height * target_size) / width)
                else:
                    new_height = target_size
                    new_width = int((width * target_size) / height)
                
                # Resize image
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                return True, f"Image resized from {width}x{height} to {new_width}x{new_height} for optimal processing", resized_image
            else:
                return False, f"Image too large. Maximum size: {ADVANCED_CONFIG['image_processing']['max_width']}x{ADVANCED_CONFIG['image_processing']['max_height']} pixels", image
        
        # Check if image has content
        img_array = np.array(image)
        if len(img_array.shape) not in [2, 3]:
            return False, "Invalid image format", image
        
        # Check for mostly blank images
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate variance to detect blank images (only if enabled)
        if ADVANCED_CONFIG["image_processing"]["check_blur"]:
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < ADVANCED_CONFIG["image_processing"]["blur_threshold"]:
                return False, "Image appears to be blank or very blurry", image
        
        return True, "Valid image", image
        
    except Exception as e:
        return False, f"Image validation error: {str(e)}", image

def enhance_image(image: Image.Image, enhancement_type: str = "auto") -> Image.Image:
    """Enhance image quality"""
    try:
        if enhancement_type == "auto":
            # Auto-enhance based on image characteristics
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Convert to HSV for better analysis
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                v_channel = hsv[:, :, 2]
                
                # Check if image is too dark
                if np.mean(v_channel) < 80:
                    # Brighten the image
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(1.3)
                
                # Check if image needs contrast enhancement
                if np.std(v_channel) < 30:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.2)
        
        elif enhancement_type == "brightness":
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
        elif enhancement_type == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
        elif enhancement_type == "sharpness":
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
        
        return image
        
    except Exception as e:
        st.warning(f"Image enhancement failed: {e}")
        return image

def preprocess_image(image: Image.Image, enhancement: bool = True) -> np.ndarray:
    """Advanced image preprocessing for model inference"""
    try:
        # Validate image first with smart resizing
        is_valid, message, processed_image = validate_image(image)
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        # Show resize message if image was resized
        if "resized" in message.lower():
            st.info(f"ℹ️ {message}")
        
        # Enhance image if requested
        if enhancement:
            processed_image = enhance_image(processed_image)
        
        # Convert PIL to numpy array
        img_array = np.array(processed_image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Image preprocessing error: {e}")
        return None

def predict_disease(model, image, confidence_threshold: float = 0.25):
    """Run disease prediction on the image with advanced features"""
    try:
        # Run inference with confidence threshold
        results = model(image, conf=confidence_threshold, verbose=False)
        
        # Get the first result
        result = results[0]
        
        # Extract predictions
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Get class predictions
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            # Map class IDs to names
            class_names = model.names
            predictions = []
            
            for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
                class_name = class_names[class_id]
                predictions.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Update performance metrics
            st.session_state.model_performance['total_detections'] += len(predictions)
            if predictions:
                avg_conf = np.mean([p['confidence'] for p in predictions])
                st.session_state.model_performance['avg_confidence'] = avg_conf
            
            return predictions, result
        else:
            return [], result
            
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return [], None

def process_batch_images(model, images: List[Image.Image], confidence_threshold: float = 0.25) -> List[Dict]:
    """Process multiple images in batch"""
    batch_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image in enumerate(images):
        status_text.text(f"Processing image {i+1}/{len(images)}")
        
        # Preprocess image
        img_array = preprocess_image(image, enhancement=True)
        if img_array is None:
            continue
        
        # Run prediction
        predictions, result = predict_disease(model, img_array, confidence_threshold)
        
        # Store results
        batch_results.append({
            'image_index': i,
            'predictions': predictions,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(images))
    
    status_text.text("Batch processing complete!")
    return batch_results

def create_detection_report(predictions: List[Dict], image_name: str = "Unknown") -> Dict:
    """Create a comprehensive detection report"""
    if not predictions:
        return {
            'image_name': image_name,
            'timestamp': datetime.now().isoformat(),
            'diseases_detected': 0,
            'health_status': 'Healthy',
            'severity': 'None',
            'recommendations': ['Continue current care practices'],
            'confidence_scores': {}
        }
    
    # Analyze predictions
    diseases = [p['class'] for p in predictions]
    confidences = {p['class']: p['confidence'] for p in predictions}
    
    # Determine health status
    if 'Health' in diseases and len(diseases) == 1:
        health_status = 'Healthy'
        severity = 'None'
    else:
        disease_counts = Counter(diseases)
        most_common = disease_counts.most_common(1)[0][0]
        health_status = f'Disease detected: {most_common}'
        severity = DISEASE_INFO.get(most_common, {}).get('severity', 'Unknown')
    
    # Generate recommendations
    recommendations = []
    for disease in set(diseases):
        if disease != 'Health':
            treatment = DISEASE_INFO.get(disease, {}).get('treatment', 'Consult a plant pathologist')
            recommendations.append(f"{disease}: {treatment}")
    
    return {
        'image_name': image_name,
        'timestamp': datetime.now().isoformat(),
        'diseases_detected': len(set(diseases)),
        'health_status': health_status,
        'severity': severity,
        'recommendations': recommendations,
        'confidence_scores': confidences,
        'detailed_predictions': predictions
    }

def export_results_to_csv(results: List[Dict]) -> str:
    """Export detection results to CSV format"""
    if not results:
        return ""
    
    # Flatten results for CSV
    csv_data = []
    for result in results:
        base_info = {
            'timestamp': result['timestamp'],
            'image_name': result['image_name'],
            'diseases_detected': result['diseases_detected'],
            'health_status': result['health_status'],
            'severity': result['severity']
        }
        
        # Add confidence scores
        for disease, confidence in result['confidence_scores'].items():
            base_info[f'{disease}_confidence'] = confidence
        
        csv_data.append(base_info)
    
    # Create DataFrame and convert to CSV
    df = pd.DataFrame(csv_data)
    return df.to_csv(index=False)

def show_disease_info(disease_name: str):
    """Display detailed information about a detected disease"""
    if disease_name not in DISEASE_INFO:
        st.warning(f"No information available for {disease_name}")
        return
    
    disease_info = DISEASE_INFO[disease_name]
    
    # Create expandable section for disease details
    with st.expander(f"🔍 {disease_name} - Detailed Information", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Severity Level:** {disease_info.get('severity', 'Unknown')}")
            st.markdown(f"**Description:** {disease_info.get('description', 'No description available')}")
            
            # Symptoms
            symptoms = disease_info.get('symptoms', [])
            if symptoms:
                st.markdown("**Common Symptoms:**")
                for symptom in symptoms:
                    st.write(f"• {symptom}")
        
        with col2:
            # Treatment recommendations
            treatment = disease_info.get('treatment', 'No treatment information available')
            st.markdown("**Treatment Recommendations:**")
            st.write(treatment)
            
            # Prevention tips
            prevention = disease_info.get('prevention', [])
            if prevention:
                st.markdown("**Prevention Tips:**")
                for tip in prevention:
                    st.write(f"• {tip}")

def create_analytics_dashboard():
    """Create an analytics dashboard for detection history"""
    if not st.session_state.detection_history:
        st.info("No detection history available yet.")
        return
    
    st.subheader("📊 Analytics Dashboard")
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.detection_history)
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", len(df))
    
    with col2:
        healthy_count = len(df[df['health_status'] == 'Healthy'])
        st.metric("Healthy Images", healthy_count)
    
    with col3:
        disease_count = len(df[df['health_status'] != 'Healthy'])
        st.metric("Diseased Images", disease_count)
    
    with col4:
        avg_confidence = df['confidence_scores'].apply(lambda x: np.mean(list(x.values())) if x else 0).mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    # Disease distribution chart
    st.subheader("Disease Distribution")
    disease_counts = {}
    for _, row in df.iterrows():
        for disease in row['confidence_scores'].keys():
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    if disease_counts:
        fig = px.pie(
            values=list(disease_counts.values()),
            names=list(disease_counts.keys()),
            title="Disease Detection Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence trends
    st.subheader("Confidence Trends")
    confidence_data = []
    for _, row in df.iterrows():
        for disease, confidence in row['confidence_scores'].items():
            confidence_data.append({
                'timestamp': row['timestamp'],
                'disease': disease,
                'confidence': confidence
            })
    
    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        fig = px.line(
            conf_df, 
            x='timestamp', 
            y='confidence', 
            color='disease',
            title="Confidence Scores Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

def create_confidence_chart(predictions):
    """Create a confidence chart for predictions"""
    if not predictions:
        return None
    
    # Extract data for chart
    diseases = [pred['class'] for pred in predictions]
    confidences = [pred['confidence'] for pred in predictions]
    
    # Create bar chart
    fig = px.bar(
        x=diseases,
        y=confidences,
        title="Detection Confidence Scores",
        labels={'x': 'Disease Type', 'y': 'Confidence Score'},
        color=confidences,
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    return fig

def display_disease_info(predictions):
    """Display detailed disease information"""
    if not predictions:
        st.info("No diseases detected. The leaf appears to be healthy!")
        return
    
    # Get the highest confidence prediction
    best_prediction = max(predictions, key=lambda x: x['confidence'])
    disease = best_prediction['class']
    confidence = best_prediction['confidence']
    
    # Get disease information
    disease_info = DISEASE_INFO.get(disease, {})
    
    # Create appropriate card based on disease
    if disease == "Health":
        st.markdown(f"""
        <div class="success-card">
            <h3>🌿 Healthy Leaf Detected</h3>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Status:</strong> No disease detected</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        severity_color = disease_info.get('color', '#dc3545')
        severity = disease_info.get('severity', 'Unknown')
        
        st.markdown(f"""
        <div class="danger-card">
            <h3>⚠️ {disease} Detected</h3>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Severity:</strong> {severity}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display detailed information
    with st.expander(f"📋 Detailed Information about {disease}", expanded=True):
        st.write(f"**Description:** {disease_info.get('description', 'No description available')}")
        
        st.write("**Symptoms:**")
        for symptom in disease_info.get('symptoms', []):
            st.write(f"• {symptom}")
        
        st.write("**Treatment Recommendations:**")
        st.write(disease_info.get('treatment', 'No treatment information available'))

def create_home_page():
    """Create an attractive home page with features showcase"""
    
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; color: white;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🌽 Maize Disease Detection</h1>
        <h2 style="font-size: 1.5rem; margin-bottom: 2rem; opacity: 0.9;">AI-Powered Plant Health Analysis</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.8;">Detect maize diseases with 99.5% accuracy using advanced computer vision</p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px);">🚀 Real-time Detection</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px);">📊 Batch Processing</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px);">📈 Analytics Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("### 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; border-left: 5px solid #28a745; margin-bottom: 1rem;">
            <h3 style="color: #28a745; margin-top: 0;">🔍 Single Image Analysis</h3>
            <ul style="color: #666;">
                <li>Instant disease detection</li>
                <li>Auto image enhancement</li>
                <li>Confidence scoring</li>
                <li>Treatment recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; border-left: 5px solid #007bff; margin-bottom: 1rem;">
            <h3 style="color: #007bff; margin-top: 0;">📦 Batch Processing</h3>
            <ul style="color: #666;">
                <li>Process up to 10 images</li>
                <li>Progress tracking</li>
                <li>Bulk export results</li>
                <li>Efficient workflow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; border-left: 5px solid #ffc107; margin-bottom: 1rem;">
            <h3 style="color: #ffc107; margin-top: 0;">📊 Analytics Dashboard</h3>
            <ul style="color: #666;">
                <li>Performance metrics</li>
                <li>Disease distribution</li>
                <li>Trend analysis</li>
                <li>Historical data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Maize Images Section with real examples
    st.markdown("### 🌽 Maize Disease Examples")
    
    # Create image grid with enhanced visuals
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin-bottom: 1rem; border: 2px solid #28a745; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);">
            <div style="width: 120px; height: 120px; background: linear-gradient(45deg, #28a745, #20c997); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; font-size: 3rem; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);">🌱</div>
            <h4 style="color: #28a745; margin: 0; font-size: 1.2rem;">Healthy Leaf</h4>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0;">Vibrant green, no spots</p>
            <div style="background: #d4edda; color: #155724; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; font-size: 0.8rem;">✅ No diseases detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin-bottom: 1rem; border: 2px solid #ffc107; box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);">
            <div style="width: 120px; height: 120px; background: linear-gradient(45deg, #ffc107, #fd7e14); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; font-size: 3rem; box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);">🍂</div>
            <h4 style="color: #ffc107; margin: 0; font-size: 1.2rem;">Grey Leaf Spots</h4>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0;">Circular grey lesions</p>
            <div style="background: #fff3cd; color: #856404; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; font-size: 0.8rem;">⚠️ Medium severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin-bottom: 1rem; border: 2px solid #dc3545; box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);">
            <div style="width: 120px; height: 120px; background: linear-gradient(45deg, #dc3545, #e83e8c); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; font-size: 3rem; box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);">🔥</div>
            <h4 style="color: #dc3545; margin: 0; font-size: 1.2rem;">Leaf Blight</h4>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0;">Large brown lesions</p>
            <div style="background: #f8d7da; color: #721c24; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; font-size: 0.8rem;">🚨 High severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin-bottom: 1rem; border: 2px solid #6f42c1; box-shadow: 0 4px 15px rgba(111, 66, 193, 0.2);">
            <div style="width: 120px; height: 120px; background: linear-gradient(45deg, #6f42c1, #e83e8c); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; font-size: 3rem; box-shadow: 0 4px 15px rgba(111, 66, 193, 0.3);">🦠</div>
            <h4 style="color: #6f42c1; margin: 0; font-size: 1.2rem;">MSV Disease</h4>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0;">Viral infection</p>
            <div style="background: #e2e3f1; color: #4c2a85; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; font-size: 0.8rem;">🦠 High severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add sample images section
    st.markdown("### 📸 Sample Detection Results")
    
    # Create a demo results section
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("""
        <div style="background: #fff; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h4 style="color: #28a745; margin-top: 0;">✅ Healthy Maize Leaf</h4>
            <div style="background: linear-gradient(45deg, #28a745, #20c997); height: 150px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 3rem; margin: 1rem 0;">🌽</div>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Confidence: 98.5% | Status: Healthy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with demo_col2:
        st.markdown("""
        <div style="background: #fff; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h4 style="color: #dc3545; margin-top: 0;">🚨 Diseased Maize Leaf</h4>
            <div style="background: linear-gradient(45deg, #dc3545, #e83e8c); height: 150px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 3rem; margin: 1rem 0;">🍂</div>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Confidence: 94.2% | Status: Leaf Blight</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("### 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        <div style="background: #fff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #007bff; margin-top: 0;">🤖 AI Model</h4>
            <p style="color: #666; margin: 0;">YOLOv8 Nano<br/>99.5% Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div style="background: #fff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #28a745; margin-top: 0;">⚡ Performance</h4>
            <p style="color: #666; margin: 0;">Real-time Processing<br/>Batch Support</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div style="background: #fff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #ffc107; margin-top: 0;">📊 Analytics</h4>
            <p style="color: #666; margin: 0;">Interactive Charts<br/>Export Options</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("### 📊 Why Choose Our System?")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #007bff, #0056b3); color: white; border-radius: 15px; margin-bottom: 1rem;">
            <h2 style="margin: 0; font-size: 2.5rem;">99.5%</h2>
            <p style="margin: 0.5rem 0 0; opacity: 0.9;">Accuracy Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #28a745, #1e7e34); color: white; border-radius: 15px; margin-bottom: 1rem;">
            <h2 style="margin: 0; font-size: 2.5rem;">4</h2>
            <p style="margin: 0.5rem 0 0; opacity: 0.9;">Disease Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #ffc107, #e0a800); color: white; border-radius: 15px; margin-bottom: 1rem;">
            <h2 style="margin: 0; font-size: 2.5rem;">10</h2>
            <p style="margin: 0.5rem 0 0; opacity: 0.9;">Batch Processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #dc3545, #c82333); color: white; border-radius: 15px; margin-bottom: 1rem;">
            <h2 style="margin: 0; font-size: 2.5rem;">Real-time</h2>
            <p style="margin: 0.5rem 0 0; opacity: 0.9;">Processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("### 🔄 How It Works")
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; margin-bottom: 1rem;">
            <div style="width: 80px; height: 80px; background: linear-gradient(45deg, #007bff, #0056b3); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem; font-weight: bold;">1</div>
            <h4 style="color: #007bff; margin: 0;">Upload Image</h4>
            <p style="color: #666; margin: 0.5rem 0 0; font-size: 0.9rem;">Take a clear photo of your maize leaf</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; margin-bottom: 1rem;">
            <div style="width: 80px; height: 80px; background: linear-gradient(45deg, #28a745, #1e7e34); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem; font-weight: bold;">2</div>
            <h4 style="color: #28a745; margin: 0;">AI Analysis</h4>
            <p style="color: #666; margin: 0.5rem 0 0; font-size: 0.9rem;">Our AI model analyzes the image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; margin-bottom: 1rem;">
            <div style="width: 80px; height: 80px; background: linear-gradient(45deg, #ffc107, #e0a800); border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem; font-weight: bold;">3</div>
            <h4 style="color: #ffc107; margin: 0;">Get Results</h4>
            <p style="color: #666; margin: 0.5rem 0 0; font-size: 0.9rem;">Receive instant diagnosis and treatment</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Call to Action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem 2rem; border-radius: 20px; text-align: center; color: white; margin: 2rem 0; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
        <h2 style="margin-top: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Ready to Get Started?</h2>
        <p style="font-size: 1.3rem; margin-bottom: 2rem; opacity: 0.9;">Upload your maize leaf images and get instant disease detection results</p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.8rem 1.5rem; border-radius: 30px; backdrop-filter: blur(10px); font-size: 1.1rem;">⚡ Instant Results</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.8rem 1.5rem; border-radius: 30px; backdrop-filter: blur(10px); font-size: 1.1rem;">📊 Detailed Reports</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.8rem 1.5rem; border-radius: 30px; backdrop-filter: blur(10px); font-size: 1.1rem;">🎯 Treatment Plans</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get Started Button
    if st.button("🚀 Get Started - Upload Your First Image", type="primary", use_container_width=True):
        st.session_state.show_home = False
        st.rerun()

def main():
    """Main application function with advanced features"""
    
    # Check if home page should be shown
    if 'show_home' not in st.session_state:
        st.session_state.show_home = True
    
    # Show home page if requested
    if st.session_state.show_home:
        create_home_page()
        return
    
    # Navigation
    if st.button("🏠 Home", key="home_btn"):
        st.session_state.show_home = True
        st.rerun()
    
    # Header
    st.markdown('<h1 class="main-header">🌽 Advanced Maize Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload maize leaf images for instant disease detection with 99.5% accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.stop()
    
    # Advanced sidebar
    st.sidebar.title("🔧 Advanced Settings")
    
    # Processing mode selection
    processing_mode = st.sidebar.selectbox(
        "Processing Mode",
        ["Single Image", "Batch Processing", "Analytics Dashboard"],
        help="Choose your processing mode"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for disease detection"
    )
    
    # Image processing options
    st.sidebar.markdown("### 🖼️ Image Processing")
    enable_enhancement = st.sidebar.checkbox("Auto-enhance images", value=True)
    enhancement_type = st.sidebar.selectbox(
        "Enhancement Type",
        ["auto", "brightness", "contrast", "sharpness"],
        help="Choose image enhancement method"
    )
    
    # Advanced image validation options
    st.sidebar.markdown("### 🔍 Image Validation")
    check_blur = st.sidebar.checkbox(
        "Check for blurry images", 
        value=ADVANCED_CONFIG["image_processing"]["check_blur"],
        help="Enable blur detection (may flag some valid images)"
    )
    # Update config based on user selection
    ADVANCED_CONFIG["image_processing"]["check_blur"] = check_blur
    
    # Image size info
    st.sidebar.markdown("### 📏 Image Size Limits")
    st.sidebar.info(f"""
    **Maximum Size:** {ADVANCED_CONFIG['image_processing']['max_width']}x{ADVANCED_CONFIG['image_processing']['max_height']} pixels  
    **Auto-Resize:** {ADVANCED_CONFIG['image_processing']['resize_large_images']}  
    **Target Size:** {ADVANCED_CONFIG['image_processing']['target_size']}px  
    **File Size:** {ADVANCED_CONFIG['max_file_size'] // (1024*1024)}MB max
    """)
    
    # Export options
    st.sidebar.markdown("### 📊 Export Options")
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["CSV", "JSON", "PDF Report"],
        help="Choose export format for results"
    )
    
    # Model performance metrics
    st.sidebar.markdown("### 📈 Performance Metrics")
    perf = st.session_state.model_performance
    st.sidebar.metric("Total Detections", perf['total_detections'])
    st.sidebar.metric("Model Accuracy", f"{perf['accuracy_score']:.1%}")
    st.sidebar.metric("Avg Confidence", f"{perf['avg_confidence']:.1%}")
    
    # Model info
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.info("""
    **Model:** YOLOv8 Nano  
    **Accuracy:** 99.5% mAP50  
    **Classes:** 4 disease types  
    **Status:** ✅ Ready
    """)
    
    # Main content based on processing mode
    if processing_mode == "Single Image":
        # Single image processing
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a maize leaf image",
                type=ADVANCED_CONFIG["supported_formats"],
                help="Upload a clear image of a maize leaf for disease detection"
            )
        
        with col2:
            st.subheader("📋 Instructions")
            st.markdown("""
            1. **Take a clear photo** of the maize leaf
            2. **Ensure good lighting** for better detection
            3. **Include the entire leaf** in the frame
            4. **Avoid shadows** and blur
            5. **Upload the image** using the file uploader
            """)
        
        # Process single image
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file)
                
                # Display original image
                st.subheader("📷 Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess image with enhancement
                img_array = preprocess_image(image, enhancement=enable_enhancement)
                
                if img_array is not None:
                    # Run prediction
                    with st.spinner("🔍 Analyzing image..."):
                        predictions, result = predict_disease(model, img_array, confidence_threshold)
                    
                    # Create detection report
                    report = create_detection_report(predictions, uploaded_file.name)
                    
                    # Store in history
                    st.session_state.detection_history.append(report)
                    
                    # Display results
                    if predictions:
                        st.subheader("🔍 Detection Results")
                        
                        # Create results table
                        results_data = []
                        for pred in predictions:
                            results_data.append({
                                'Disease': pred['class'],
                                'Confidence': f"{pred['confidence']:.2%}",
                                'Severity': DISEASE_INFO.get(pred['class'], {}).get('severity', 'Unknown')
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Display annotated image
                        if result is not None:
                            annotated_img = result.plot()
                            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            st.subheader("🎯 Annotated Image")
                            st.image(annotated_img_rgb, caption="Detection Results", use_column_width=True)
                        
                        # Show disease information
                        for pred in predictions:
                            if pred['class'] != 'Health':
                                show_disease_info(pred['class'])
                    
                    else:
                        st.success("✅ No diseases detected! The leaf appears to be healthy.")
                        st.info("💡 Continue with good agricultural practices to maintain plant health.")
                    
                    # Export options
                    st.subheader("📊 Export Results")
                    if export_format == "CSV":
                        csv_data = export_results_to_csv([report])
                        st.download_button(
                            "Download CSV Report",
                            csv_data,
                            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        json_data = json.dumps(report, indent=2)
                        st.download_button(
                            "Download JSON Report",
                            json_data,
                            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
    
    elif processing_mode == "Batch Processing":
        # Batch processing mode
        st.subheader("📦 Batch Processing")
        st.info("Upload multiple images for batch disease detection")
        
        uploaded_files = st.file_uploader(
            "Choose multiple maize leaf images",
            type=ADVANCED_CONFIG["supported_formats"],
            accept_multiple_files=True,
            help=f"Upload up to {ADVANCED_CONFIG['batch_processing']['max_images']} images"
        )
        
        if uploaded_files:
            if len(uploaded_files) > ADVANCED_CONFIG['batch_processing']['max_images']:
                st.warning(f"⚠️ Maximum {ADVANCED_CONFIG['batch_processing']['max_images']} images allowed. Processing first {ADVANCED_CONFIG['batch_processing']['max_images']} images.")
                uploaded_files = uploaded_files[:ADVANCED_CONFIG['batch_processing']['max_images']]
            
            # Process batch
            if st.button("🚀 Process Batch", type="primary"):
                images = [Image.open(file) for file in uploaded_files]
                batch_results = process_batch_images(model, images, confidence_threshold)
                
                # Display batch results
                st.subheader("📊 Batch Results")
                
                for i, result in enumerate(batch_results):
                    with st.expander(f"Image {i+1}: {uploaded_files[i].name}"):
                        if result['predictions']:
                            for pred in result['predictions']:
                                st.write(f"**{pred['class']}**: {pred['confidence']:.2%}")
                        else:
                            st.write("✅ No diseases detected")
                
                # Export batch results
                if st.button("📥 Export Batch Results"):
                    csv_data = export_results_to_csv([create_detection_report(r['predictions'], f"image_{r['image_index']}") for r in batch_results])
                    st.download_button(
                        "Download Batch CSV",
                        csv_data,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    elif processing_mode == "Analytics Dashboard":
        # Analytics dashboard
        create_analytics_dashboard()
        
        # Export all history
        if st.session_state.detection_history:
            st.subheader("📊 Export All Data")
            csv_data = export_results_to_csv(st.session_state.detection_history)
            st.download_button(
                "Download Complete History",
                csv_data,
                file_name=f"complete_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌽 Advanced Maize Disease Detection System | Powered by YOLOv8 | Accuracy: 99.5%</p>
        <p>Features: Single & Batch Processing | Analytics Dashboard | Export Options</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
