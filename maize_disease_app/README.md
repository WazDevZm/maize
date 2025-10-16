# 🌽 Maize Disease Detection System

An advanced AI-powered web application for detecting maize leaf diseases using YOLOv8 computer vision technology.

## 🚀 Features

- **High Accuracy**: 99.5% mAP50 accuracy in disease detection
- **Real-time Analysis**: Instant disease classification and diagnosis
- **Multiple Disease Support**: Detects 4 major maize diseases
- **Smart Recommendations**: AI-powered treatment and prevention advice
- **User-friendly Interface**: Intuitive web application with navigation
- **Model Performance Metrics**: Detailed model information and performance data

## 🔬 Supported Diseases

1. **🌿 Healthy Leaves** - Detection of healthy maize leaves with no disease symptoms
2. **🟡 Grey Leaf Spots** - Cercospora zeae-maydis infection detection
3. **🔴 Leaf Blight** - Helminthosporium maydis disease identification
4. **🦠 Maize Streak Virus** - Viral infection detection and analysis

## 🛠️ Technology Stack

- **AI Model**: YOLOv8 (You Only Look Once version 8)
- **Framework**: Ultralytics YOLO
- **Web Interface**: Streamlit
- **Computer Vision**: OpenCV, PIL
- **Visualization**: Plotly
- **Backend**: Python 3.8+

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd maize_disease_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure trained models are available**:
   - The app automatically finds and loads the best available trained model
   - Models should be located in `../dataset_split/runs/train_*/weights/best.pt`

## 🚀 Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate the interface**:
   - **🏠 Home**: Introduction and quick start
   - **🔍 Disease Detection**: Upload images and analyze diseases
   - **📊 Model Info**: View model performance metrics
   - **ℹ️ About**: Learn more about the system

3. **Upload and analyze**:
   - Upload a clear image of a maize leaf
   - Click "Analyze Disease" to get instant results
   - View detailed disease information and treatment recommendations

## 🎯 Model Loading

The application automatically:
- Searches for available trained models
- Loads the most recent model by default
- Displays model performance metrics
- Provides fallback error handling

## 📊 Performance Metrics

The system displays comprehensive model information including:
- Model accuracy (mAP50, mAP50-95)
- Precision and recall metrics
- Model size and last update time
- Training performance data

## 🔧 Configuration

- **Confidence Threshold**: Adjustable detection sensitivity (0.1-1.0)
- **Model Selection**: Automatic best model detection
- **Image Formats**: Supports JPG, JPEG, PNG, BMP

## 🌱 Agricultural Benefits

- **Early Disease Detection**: Identify diseases before they spread
- **Treatment Recommendations**: Get specific treatment advice
- **Crop Health Monitoring**: Regular monitoring capabilities
- **Sustainable Agriculture**: Support for sustainable farming practices

## 📈 Future Enhancements

- Support for additional crop diseases
- Mobile application development
- Integration with IoT sensors
- Advanced treatment recommendations
- Multi-language support

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For technical support, feature requests, or collaboration opportunities, please reach out through our development team.

---

**🌽 Maize Disease Detection System**  
*Empowering Agriculture with AI Technology*