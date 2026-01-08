"""
Server startup script for Maize Disease Detection API
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Get the current directory
    current_dir = Path(__file__).parent
    
    print("🌽 Starting Maize Disease Detection API Server...")
    print(f"📁 Server directory: {current_dir}")
    print("🔗 API will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🔄 Health check: http://localhost:8000/health")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )