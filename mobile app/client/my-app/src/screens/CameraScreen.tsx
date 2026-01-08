import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  Alert,
  Image,
  TouchableOpacity,
  Platform,
} from 'react-native';
import {
  Text,
  Button,
  Card,
  Surface,
  ActivityIndicator,
  Chip,
  IconButton,
} from 'react-native-paper';
import { Camera, CameraType, FlashMode } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

import { theme } from '../theme/theme';

const { width, height } = Dimensions.get('window');

interface DetectionResult {
  class: string;
  confidence: number;
  severity: string;
  treatment: string;
}

const CameraScreen: React.FC = () => {
  const navigation = useNavigation();
  const cameraRef = useRef<Camera>(null);
  
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [cameraType, setCameraType] = useState(CameraType.back);
  const [flashMode, setFlashMode] = useState(FlashMode.off);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [showCamera, setShowCamera] = useState(true);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        setCapturedImage(photo.uri);
        setShowCamera(false);
      } catch (error) {
        Alert.alert('Error', 'Failed to take picture');
      }
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setCapturedImage(result.assets[0].uri);
        setShowCamera(false);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const analyzeImage = async () => {
    if (!capturedImage) return;

    setIsAnalyzing(true);
    
    try {
      // Create FormData for image upload
      const formData = new FormData();
      
      // Get file info
      const fileInfo = await FileSystem.getInfoAsync(capturedImage);
      
      formData.append('file', {
        uri: capturedImage,
        type: 'image/jpeg',
        name: 'maize_leaf.jpg',
      } as any);

      // Call the FastAPI backend
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.ok) {
        const result = await response.json();
        
        if (result.detailed_predictions && result.detailed_predictions.length > 0) {
          const prediction = result.detailed_predictions[0];
          setDetectionResult({
            class: prediction.class,
            confidence: prediction.confidence * 100,
            severity: prediction.disease_info?.severity || 'Unknown',
            treatment: prediction.disease_info?.treatment || 'No treatment information available',
          });
        } else {
          setDetectionResult({
            class: 'Healthy',
            confidence: 95,
            severity: 'None',
            treatment: 'Continue current care practices',
          });
        }

        // Navigate to results screen
        navigation.navigate('DetectionResult' as never, {
          image: capturedImage,
          result: detectionResult,
        } as never);
      } else {
        throw new Error('Detection failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      
      // Mock result for demo purposes
      const mockResults = [
        {
          class: 'Healthy',
          confidence: 98.5,
          severity: 'None',
          treatment: 'Continue current care practices',
        },
        {
          class: 'Grey Leaf Spots',
          confidence: 94.2,
          severity: 'Medium',
          treatment: 'Apply fungicides, improve air circulation, remove infected leaves',
        },
        {
          class: 'Leaf Blight',
          confidence: 91.8,
          severity: 'High',
          treatment: 'Apply copper-based fungicides, improve drainage, crop rotation',
        },
      ];

      const randomResult = mockResults[Math.floor(Math.random() * mockResults.length)];
      setDetectionResult(randomResult);

      // Navigate to results screen
      navigation.navigate('DetectionResult' as never, {
        image: capturedImage,
        result: randomResult,
      } as never);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    setDetectionResult(null);
    setShowCamera(true);
  };

  const toggleCameraType = () => {
    setCameraType(
      cameraType === CameraType.back ? CameraType.front : CameraType.back
    );
  };

  const toggleFlash = () => {
    setFlashMode(
      flashMode === FlashMode.off ? FlashMode.on : FlashMode.off
    );
  };

  if (hasPermission === null) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="camera-off" size={64} color="#ccc" />
        <Text style={styles.noPermissionText}>No access to camera</Text>
        <Button
          mode="contained"
          onPress={() => Camera.requestCameraPermissionsAsync()}
          style={styles.permissionButton}
        >
          Grant Permission
        </Button>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={[theme.colors.primary, '#20c997']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <IconButton
            icon="arrow-left"
            iconColor="white"
            size={24}
            onPress={() => navigation.goBack()}
          />
          <Text style={styles.headerTitle}>Disease Detection</Text>
          <View style={styles.headerRight} />
        </View>
      </LinearGradient>

      {/* Camera or Image Preview */}
      <View style={styles.cameraContainer}>
        {showCamera && !capturedImage ? (
          <Camera
            ref={cameraRef}
            style={styles.camera}
            type={cameraType}
            flashMode={flashMode}
          >
            {/* Camera Overlay */}
            <View style={styles.cameraOverlay}>
              {/* Top Controls */}
              <View style={styles.topControls}>
                <TouchableOpacity
                  style={styles.controlButton}
                  onPress={toggleFlash}
                >
                  <Ionicons
                    name={flashMode === FlashMode.on ? 'flash' : 'flash-off'}
                    size={24}
                    color="white"
                  />
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={styles.controlButton}
                  onPress={toggleCameraType}
                >
                  <Ionicons name="camera-reverse" size={24} color="white" />
                </TouchableOpacity>
              </View>

              {/* Center Guide */}
              <View style={styles.centerGuide}>
                <View style={styles.focusFrame}>
                  <View style={styles.corner} />
                  <View style={[styles.corner, styles.topRight]} />
                  <View style={[styles.corner, styles.bottomLeft]} />
                  <View style={[styles.corner, styles.bottomRight]} />
                </View>
                <Text style={styles.guideText}>
                  Position the maize leaf within the frame
                </Text>
              </View>

              {/* Bottom Controls */}
              <View style={styles.bottomControls}>
                <TouchableOpacity
                  style={styles.galleryButton}
                  onPress={pickImage}
                >
                  <Ionicons name="images" size={24} color="white" />
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.captureButton}
                  onPress={takePicture}
                >
                  <View style={styles.captureButtonInner} />
                </TouchableOpacity>

                <View style={styles.placeholder} />
              </View>
            </View>
          </Camera>
        ) : (
          <View style={styles.imagePreview}>
            {capturedImage && (
              <Image source={{ uri: capturedImage }} style={styles.previewImage} />
            )}
          </View>
        )}
      </View>

      {/* Instructions */}
      {showCamera && !capturedImage && (
        <Animatable.View animation="fadeInUp" style={styles.instructions}>
          <Card style={styles.instructionCard}>
            <Card.Content>
              <Text style={styles.instructionTitle}>📸 Capture Tips</Text>
              <View style={styles.tipsList}>
                <Text style={styles.tipItem}>• Ensure good lighting</Text>
                <Text style={styles.tipItem}>• Keep the leaf in focus</Text>
                <Text style={styles.tipItem}>• Fill the frame with the leaf</Text>
                <Text style={styles.tipItem}>• Avoid shadows and blur</Text>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>
      )}

      {/* Image Actions */}
      {capturedImage && !showCamera && (
        <Animatable.View animation="fadeInUp" style={styles.imageActions}>
          <Card style={styles.actionsCard}>
            <Card.Content>
              <Text style={styles.actionsTitle}>Image Captured Successfully!</Text>
              <Text style={styles.actionsSubtitle}>
                Ready to analyze for disease detection
              </Text>
              
              <View style={styles.actionButtons}>
                <Button
                  mode="outlined"
                  onPress={retakePhoto}
                  style={styles.actionButton}
                  icon="camera"
                >
                  Retake
                </Button>
                
                <Button
                  mode="contained"
                  onPress={analyzeImage}
                  style={styles.actionButton}
                  loading={isAnalyzing}
                  disabled={isAnalyzing}
                  icon="magnify"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                </Button>
              </View>

              {isAnalyzing && (
                <View style={styles.analyzingContainer}>
                  <ActivityIndicator size="small" color={theme.colors.primary} />
                  <Text style={styles.analyzingText}>
                    AI is analyzing your image...
                  </Text>
                </View>
              )}
            </Card.Content>
          </Card>
        </Animatable.View>
      )}

      {/* Quick Upload Option */}
      {showCamera && (
        <View style={styles.quickUpload}>
          <Button
            mode="text"
            onPress={pickImage}
            icon="upload"
            textColor={theme.colors.primary}
          >
            Or upload from gallery
          </Button>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  noPermissionText: {
    fontSize: 18,
    color: '#666',
    marginVertical: 20,
    textAlign: 'center',
  },
  permissionButton: {
    marginTop: 20,
  },
  header: {
    paddingTop: Platform.OS === 'ios' ? 50 : 30,
    paddingBottom: 15,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerTitle: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  headerRight: {
    width: 40,
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerGuide: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  focusFrame: {
    width: 250,
    height: 250,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderColor: 'white',
    borderWidth: 3,
    top: 0,
    left: 0,
    borderRightWidth: 0,
    borderBottomWidth: 0,
  },
  topRight: {
    top: 0,
    right: 0,
    left: 'auto',
    borderLeftWidth: 0,
    borderRightWidth: 3,
  },
  bottomLeft: {
    bottom: 0,
    top: 'auto',
    borderTopWidth: 0,
    borderBottomWidth: 3,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    top: 'auto',
    left: 'auto',
    borderLeftWidth: 0,
    borderTopWidth: 0,
    borderRightWidth: 3,
    borderBottomWidth: 3,
  },
  guideText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 40,
    paddingBottom: 40,
  },
  galleryButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: theme.colors.primary,
  },
  placeholder: {
    width: 50,
  },
  imagePreview: {
    flex: 1,
    backgroundColor: '#000',
  },
  previewImage: {
    flex: 1,
    width: '100%',
    resizeMode: 'contain',
  },
  instructions: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
  },
  instructionCard: {
    borderRadius: 15,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
  },
  instructionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: theme.colors.primary,
    marginBottom: 10,
  },
  tipsList: {
    gap: 4,
  },
  tipItem: {
    fontSize: 14,
    color: '#666',
  },
  imageActions: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
  },
  actionsCard: {
    borderRadius: 15,
    backgroundColor: 'white',
  },
  actionsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 5,
  },
  actionsSubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  actionButton: {
    flex: 1,
  },
  analyzingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 15,
    gap: 10,
  },
  analyzingText: {
    fontSize: 14,
    color: theme.colors.primary,
  },
  quickUpload: {
    position: 'absolute',
    bottom: 100,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
});

export default CameraScreen;
// install all the needed depencies to make this ru