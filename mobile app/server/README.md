# 🌽 Maize Disease Detection API Server

FastAPI backend server for the Maize Disease Detection mobile application. This server provides REST API endpoints for disease detection using a trained YOLOv8 model.

## 🚀 Features

- **Disease Detection**: Analyze maize leaf images for disease identification
- **Batch Processing**: Process multiple images simultaneously
- **High Accuracy**: 99.5% accuracy using YOLOv8 model
- **RESTful API**: Clean and documented API endpoints
- **Real-time Processing**: Fast inference and response times

## 📦 Installation

1. **Navigate to server directory**:
   ```bash
   cd "mobile app/server"
   ```

2. **Create virtual environment** (if not already created):
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure model file exists**:
   - The `best.pt` model file should be in the server directory
   - If not present, copy from: `../../dataset_split/runs/train_20251010_201550/weights/best.pt`

## 🚀 Running the Server

### Method 1: Using the startup script
```bash
python start.py
```

### Method 2: Direct uvicorn command
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## 📚 API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔗 API Endpoints

### Health Check
- **GET** `/health` - Check server and model status

### Disease Information
- **GET** `/diseases` - Get all supported diseases
- **GET** `/diseases/{disease_name}` - Get specific disease information

### Disease Detection
- **POST** `/detect` - Detect disease in single image
  - **Parameters**: 
    - `file`: Image file (multipart/form-data)
    - `confidence`: Detection confidence threshold (0.1-1.0)

- **POST** `/detect-batch` - Detect diseases in multiple images
  - **Parameters**: 
    - `files`: Multiple image files (multipart/form-data)
    - `confidence`: Detection confidence threshold (0.1-1.0)

## 🔧 Configuration

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF

### Model Information
- **Model**: YOLOv8 Nano
- **Accuracy**: 99.5% mAP50
- **Classes**: 4 (Health, Grey_Leaf_Spots, Leaf_Blight, MSV)
- **Input Size**: 640x640 pixels

### Disease Classes
1. **Health** - Healthy maize leaves
2. **Grey_Leaf_Spots** - Cercospora zeae-maydis infection
3. **Leaf_Blight** - Helminthosporium maydis disease
4. **MSV** - Maize Streak Virus

## 🛠️ Development

### Project Structure
```
server/
├── main.py              # FastAPI application
├── start.py             # Server startup script
├── requirements.txt     # Python dependencies
├── best.pt             # YOLOv8 model weights
└── README.md           # This file
```

### Adding New Features
1. Add new endpoints in `main.py`
2. Update the API documentation
3. Test with the interactive docs at `/docs`

## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure `best.pt` exists in the server directory
   - Check the model path in `load_model()` function

2. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

3. **Port already in use**:
   - Change the port in `start.py` or use: `uvicorn main:app --port 8001`

4. **CORS issues**:
   - The server allows all origins by default
   - For production, update CORS settings in `main.py`

### Performance Optimization
- Use GPU if available (update PyTorch installation)
- Adjust confidence threshold for speed vs accuracy trade-off
- Consider model quantization for faster inference

## 📱 Mobile App Integration

The server is designed to work with the React Native mobile app. Make sure to:

1. Update the API base URL in the mobile app configuration
2. Use your computer's IP address for physical device testing
3. Use appropriate emulator URLs for development

### Network Configuration
- **Android Emulator**: `http://10.0.2.2:8000`
- **iOS Simulator**: `http://localhost:8000`
- **Physical Device**: `http://YOUR_IP_ADDRESS:8000`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For technical support or questions:
- Check the API documentation at `/docs`
- Review the troubleshooting section
- Open an issue in the repository