# Maize Disease Detection Mobile App

A React Native mobile application for detecting maize diseases using AI-powered image analysis.

## Features

- 🌽 AI-powered disease detection with 99.5% accuracy
- 📸 Camera capture and image upload functionality
- 📊 Farm tracking and analytics dashboard
- 📈 Detection history and trends
- 💡 Disease information and treatment recommendations
- 🔐 User authentication system

## Demo Credentials

For testing purposes, use these dummy credentials:

### Farmer Account
- **Email:** `farmer@maize.com`
- **Password:** `password123`
- **Role:** Basic farmer account

### Farm Manager Account
- **Email:** `manager@farm.com`
- **Password:** `manager123`
- **Role:** Farm management account with extended features

### Agricultural Expert Account
- **Email:** `expert@agro.com`
- **Password:** `expert123`
- **Role:** Agricultural expert account with full access

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- Expo CLI
- React Native development environment
- Backend server running on `http://localhost:8000`

### Installation

1. Clone the repository
2. Navigate to the client directory:
   ```bash
   cd "mobile app/client/my-app"
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npx expo start
   ```

### Backend Connection

The app connects to a FastAPI backend server running on `http://localhost:8000`. Make sure the backend server is running before using the app.

#### Backend Endpoints:
- `GET /health` - Health check
- `POST /detect` - Disease detection
- `GET /diseases` - Get disease information
- `POST /detect-batch` - Batch disease detection

## App Structure

```
src/
├── screens/          # App screens
├── services/         # API and storage services
├── types/           # TypeScript type definitions
├── theme/           # App theme configuration
└── utils/           # Utility functions
```

## Key Screens

- **AuthScreen**: User authentication with dummy credentials
- **HomeScreen**: Dashboard with farm statistics and quick actions
- **CameraScreen**: Camera capture for disease detection
- **HistoryScreen**: Detection history and analytics
- **InfoScreen**: Disease information database
- **DetectionResultScreen**: Analysis results and recommendations
- **FarmTrackerScreen**: Farm management and tracking

## Farm Tracking Features

- Field management and organization
- Crop health monitoring
- Disease outbreak tracking
- Treatment history
- Yield predictions
- Weather integration

## Technology Stack

- **Frontend**: React Native with Expo
- **Navigation**: React Navigation
- **UI Components**: React Native Paper
- **State Management**: React Hooks
- **Storage**: AsyncStorage
- **Camera**: Expo Camera
- **Image Processing**: Expo Image Picker
- **Backend**: FastAPI (Python)
- **AI Model**: YOLOv8 for disease detection

## Development Notes

- The app uses TypeScript for type safety
- All navigation is properly typed
- Error handling is implemented throughout
- Offline storage for detection history
- Real-time camera preview with guides
- Batch image processing support

## Testing

Use the provided dummy credentials to test different user roles and features. The app includes mock data for demonstration purposes when the backend is not available.

## Support

For issues or questions, please refer to the project documentation or contact the development team.