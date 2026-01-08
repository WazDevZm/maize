# 🌽 Maize Disease Detector - Mobile App

A modern React Native mobile application for detecting maize leaf diseases using AI. Built with TypeScript, Expo, and React Native Paper for a professional user experience.

## 🚀 Features

- **📷 Camera Integration**: Take photos or select from gallery
- **🤖 AI Disease Detection**: Real-time analysis with 99.5% accuracy
- **📊 Detection History**: Track and review past detections
- **ℹ️ Disease Information**: Comprehensive disease database
- **🎨 Modern UI**: Beautiful interface with animations
- **📱 Cross-Platform**: Works on both iOS and Android

## 🛠️ Technology Stack

- **React Native** with **TypeScript**
- **Expo SDK 50** for development and deployment
- **React Native Paper** for Material Design components
- **React Navigation 6** for navigation
- **Expo Camera** for camera functionality
- **Expo Image Picker** for gallery access
- **AsyncStorage** for local data persistence
- **Axios** for API communication
- **React Native Animatable** for smooth animations

## 📦 Installation

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Expo CLI: `npm install -g @expo/cli`
- For iOS: Xcode (macOS only)
- For Android: Android Studio

### Setup Steps

1. **Navigate to client directory**:
   ```bash
   cd "mobile app/client"
   ```

2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure API endpoint**:
   - Open `src/config/api.ts`
   - Update `API_BASE_URL` with your server's IP address
   - For development, use:
     - Android Emulator: `http://10.0.2.2:8000`
     - iOS Simulator: `http://localhost:8000`
     - Physical Device: `http://YOUR_COMPUTER_IP:8000`

4. **Start the development server**:
   ```bash
   npm start
   # or
   expo start
   ```

## 🚀 Running the App

### Development Mode

1. **Start Expo development server**:
   ```bash
   expo start
   ```

2. **Run on device/emulator**:
   - **iOS**: Press `i` or scan QR code with Camera app
   - **Android**: Press `a` or scan QR code with Expo Go app
   - **Web**: Press `w` (limited functionality)

### Building for Production

1. **Build for Android**:
   ```bash
   expo build:android
   ```

2. **Build for iOS**:
   ```bash
   expo build:ios
   ```

## 📱 App Structure

```
src/
├── components/          # Reusable UI components
├── screens/            # Main app screens
│   ├── HomeScreen.tsx     # Welcome and overview
│   ├── CameraScreen.tsx   # Disease detection
│   ├── HistoryScreen.tsx  # Detection history
│   └── InfoScreen.tsx     # Disease information
├── types/              # TypeScript type definitions
│   ├── api.ts            # API response types
│   ├── navigation.ts     # Navigation types
│   └── index.ts          # Type exports
├── config/             # Configuration files
│   └── api.ts            # API client setup
├── utils/              # Utility functions
│   └── storage.ts        # Local storage helpers
└── theme/              # App theming
    └── theme.ts          # Material Design theme
```

## 🎨 Features Overview

### 🏠 Home Screen
- Server status monitoring
- Quick action buttons
- Feature highlights
- Supported disease overview

### 📷 Camera Screen
- Camera capture with preview
- Gallery image selection
- Real-time disease detection
- Detailed results with treatment recommendations
- Haptic feedback for better UX

### 📊 History Screen
- Detection history with search
- Statistics overview
- Individual result details
- Export and management options

### ℹ️ Info Screen
- Comprehensive disease database
- Symptoms and treatment information
- Prevention tips
- AI model information

## 🔧 Configuration

### API Configuration
Update `src/config/api.ts`:

```typescript
const API_BASE_URL = __DEV__ 
  ? 'http://10.0.2.2:8000' // Development
  : 'http://your-production-server.com'; // Production
```

### Theme Customization
Modify `src/theme/theme.ts`:

```typescript
export const theme = {
  colors: {
    primary: '#2E8B57', // Your brand color
    // ... other colors
  },
};
```

## 📱 Permissions

The app requires the following permissions:

### iOS (Info.plist)
- `NSCameraUsageDescription`: Camera access for capturing leaf images
- `NSPhotoLibraryUsageDescription`: Photo library access for image selection

### Android (AndroidManifest.xml)
- `android.permission.CAMERA`: Camera access
- `android.permission.READ_EXTERNAL_STORAGE`: Gallery access
- `android.permission.WRITE_EXTERNAL_STORAGE`: Image saving

## 🔍 API Integration

### Detection Flow
1. User captures/selects image
2. Image is sent to FastAPI server
3. YOLOv8 model processes the image
4. Results are returned with disease information
5. Results are saved to local history

### Error Handling
- Network connectivity checks
- Server availability monitoring
- Graceful error messages
- Offline capability for viewing history

## 🎯 Performance Optimization

### Image Processing
- Automatic image resizing for optimal processing
- Quality optimization for faster uploads
- Caching for better performance

### Memory Management
- Efficient image handling
- History limit (100 detections)
- Automatic cleanup of temporary files

### Network Optimization
- Request timeout handling
- Retry mechanisms
- Compression for faster transfers

## 🧪 Testing

### Running Tests
```bash
npm test
# or
yarn test
```

### Testing on Devices
1. **Physical Device**: Install Expo Go app and scan QR code
2. **iOS Simulator**: Use Xcode simulator
3. **Android Emulator**: Use Android Studio AVD

## 🚀 Deployment

### Expo Application Services (EAS)
1. **Install EAS CLI**:
   ```bash
   npm install -g eas-cli
   ```

2. **Configure EAS**:
   ```bash
   eas build:configure
   ```

3. **Build for stores**:
   ```bash
   eas build --platform all
   ```

### Standalone Apps
- Generate APK/IPA files for distribution
- Configure app signing and certificates
- Submit to Google Play Store / Apple App Store

## 🔧 Troubleshooting

### Common Issues

1. **Metro bundler issues**:
   ```bash
   expo start --clear
   ```

2. **Node modules problems**:
   ```bash
   rm -rf node_modules
   npm install
   ```

3. **iOS build issues**:
   - Check Xcode version compatibility
   - Verify iOS deployment target

4. **Android build issues**:
   - Check Android SDK versions
   - Verify Gradle configuration

### Network Issues
- Ensure server is running on correct port
- Check firewall settings
- Verify IP address configuration
- Test API endpoints manually

## 📚 Learning Resources

- [Expo Documentation](https://docs.expo.dev/)
- [React Native Documentation](https://reactnative.dev/)
- [React Navigation](https://reactnavigation.org/)
- [React Native Paper](https://reactnativepaper.com/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes with proper TypeScript types
4. Test on both iOS and Android
5. Submit a pull request

### Code Style
- Use TypeScript for all new code
- Follow React Native best practices
- Use meaningful component and variable names
- Add proper error handling
- Include JSDoc comments for complex functions

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For technical support:
- Check the troubleshooting section
- Review Expo documentation
- Open an issue in the repository
- Contact the development team

## 🔮 Future Enhancements

- [ ] Offline mode with local model
- [ ] Push notifications for detection results
- [ ] Social sharing of results
- [ ] Multi-language support
- [ ] Advanced analytics and insights
- [ ] Integration with farming management systems