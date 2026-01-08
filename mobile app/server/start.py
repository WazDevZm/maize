"""
Startup script for the Maize Disease Detection API server
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Start the FastAPI server"""
    print("🌽 Starting Maize Disease Detection API Server...")
    print("=" * 50)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Check if model files exist
    model_paths = [
        current_dir / "best.pt",
        current_dir / "../maize_disease_app/best.pt",
        current_dir / "yolov8n.pt"
    ]
    
    model_found = False
    for model_path in model_paths:
        if model_path.exists():
            print(f"✅ Model found: {model_path}")
            model_found = True
            break
    
    if not model_found:
        print("⚠️  No model file found. The server will start but may not work properly.")
        print("   Expected locations:")
        for path in model_paths:
            print(f"   - {path}")
    
    print("\n🚀 Starting server on http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()