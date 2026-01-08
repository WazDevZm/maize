import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  StyleSheet,
  Alert,
  Dimensions,
  Image,
  ScrollView,
} from 'react-native';
import {
  Button,
  Card,
  Title,
  Paragraph,
  Text,
  Surface,
  ActivityIndicator,
  Chip,
  Divider,
} from 'react-native-paper';
import { Camera, CameraType } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import { Ionicons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import * as Haptics from 'expo-haptics';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

import { theme } from '@/theme/theme';
import { apiService } from '@/config/api';
import type { RootTabParamList, DetectionResult } from '@/types';
import { saveDetectionHistory } from '@/utils/storage';

const { width, height } = Dimensions.get('window');

type Props = BottomTabScreenProps<RootTabParamList, 'Camera'>;

interface DetectionState {
  isDetecting: boolean;
  result: DetectionResult | null;
  imageUri: string | null;
  confidence: number;
}

const CameraScreen: React.FC<Props> = ({ navigation }) => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [cameraType, setCameraType] = useState(CameraType.back);
  const [showCamera, setShowCamera] = useState(false);
  const [detection, setDetection] = useState<DetectionState>({
    isDetecting: false,
    result: null,
    imageUri: null,
    confidence: 0.25,
  });
  
  const cameraRef = useRef<Camera>(null);

  useEffect(() => {
    requestPermissions();
  }, []);

  const requestPermissions = async (): Promise<void> => {
    const { status: cameraStatus } = await Camera.requestCameraPermissionsAsync();
    const { status: mediaStatus } = await MediaLibrary.requestPermissionsAsync();
    const { status: imagePickerStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    setHasPermission(
      cameraStatus === 'granted' && 
      mediaStatus === 'granted' && 
      imagePickerStatus === 'granted'
    );
  };

  const takePicture = async (): Promise<void> => {
    if (cameraRef.current) {
      try {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        
        setShowCamera(false);
        await processImage(photo.uri);
      } catch (error) {
        console.error('Error taking picture:', error);
        Alert.alert('Error', 'Failed to take picture. Please try again.');
      }
    }
  };

  const pickImage = async (): Promise<void> => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        await processImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  const processImage = async (imageUri: string): Promise<void> => {
    try {
      setDetection(prev => ({
        ...prev,
        isDetecting: true,
        imageUri,
        result: null,
      }));

      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      
      const result = await apiService.detectDisease(imageUri, detection.confidence);
      
      // Save to history
      await saveDetectionHistory({
        id: Date.now().toString(),
        imageUri,
        result,
        timestamp: new Date().toISOString(),
      });

      setDetection(prev => ({
        ...prev,
        isDetecting: false,
        result,
      }));

      await Haptics.notificationAsync(
        result.predictions.length > 0 && result.predictions.some(p => p.class !== 'Health')
          ? Haptics.NotificationFeedbackType.Warning
          : Haptics.NotificationFeedbackType.Success
      );
      
    } catch (error) {
      console.error('Detection error:', error);
      setDetection(prev => ({
        ...prev,
        isDetecting: false,
      }));
      
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      
      Alert.alert(
        'Detection Failed',
        'Unable to analyze the image. Please check your connection and try again.',
        [{ text: 'OK' }]
      );
    }
  };

  const resetDetection = (): void => {
    setDetection({
      isDetecting: false,
      result: null,
      imageUri: null,
      confidence: 0.25,
    });
  };

  const renderPermissionRequest = (): JSX.Element => (
    <View style={styles.centerContainer}>
      <Ionicons name="camera-outline" size={64} color={theme.colors.primary} />
      <Title style={styles.permissionTitle}>Camera Permission Required</Title>
      <Paragraph style={styles.permissionText}>
        This app needs camera and photo library access to detect diseases in maize leaves.
      </Paragraph>
      <Button mode="contained" onPress={requestPermissions} style={styles.permissionButton}>
        Grant Permissions
      </Button>
    </View>
  );

  const renderCameraView = (): JSX.Element => (
    <View style={styles.cameraContainer}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={cameraType}
        ratio="4:3"
      >
        <View style={styles.cameraOverlay}>
          <View style={styles.cameraHeader}>
            <Button
              mode="contained-tonal"
              onPress={() => setShowCamera(false)}
              style={styles.closeButton}
            >
              Close
            </Button>
            <Button
              mode="contained-tonal"
              onPress={() => setCameraType(
                cameraType === CameraType.back ? CameraType.front : CameraType.back
              )}
              style={styles.flipButton}
            >
              Flip
            </Button>
          </View>
          
          <View style={styles.cameraFooter}>
            <View style={styles.captureContainer}>
              <Button
                mode="contained"
                onPress={takePicture}
                style={styles.captureButton}
                contentStyle={styles.captureButtonContent}
              >
                📷 Capture
              </Button>
            </View>
          </View>
        </View>
      </Camera>
    </View>
  );

  const renderDetectionResult = (): JSX.Element | null => {
    if (!detection.result || !detection.imageUri) return null;

    const { result } = detection;
    const hasDisease = result.predictions.some(p => p.class !== 'Health');

    return (
      <ScrollView style={styles.resultContainer} showsVerticalScrollIndicator={false}>
        <Animatable.View animation="fadeInUp" duration={600}>
          {/* Image */}
          <Card style={styles.imageCard}>
            <Image source={{ uri: detection.imageUri }} style={styles.resultImage} />
          </Card>

          {/* Overall Status */}
          <Surface style={styles.statusCard} elevation={2}>
            <View style={styles.statusHeader}>
              <Ionicons 
                name={hasDisease ? "warning" : "checkmark-circle"} 
                size={32} 
                color={hasDisease ? theme.colors.warning : theme.colors.success} 
              />
              <View style={styles.statusText}>
                <Title style={styles.statusTitle}>{result.health_status}</Title>
                <Text style={styles.statusSubtitle}>
                  Severity: {result.severity} | Detections: {result.total_detections}
                </Text>
              </View>
            </View>
          </Surface>

          {/* Predictions */}
          {result.detailed_predictions.length > 0 && (
            <Card style={styles.predictionsCard}>
              <Card.Content>
                <Title>Detection Results</Title>
                {result.detailed_predictions.map((prediction, index) => (
                  <View key={index} style={styles.predictionItem}>
                    <View style={styles.predictionHeader}>
                      <Chip 
                        style={[
                          styles.diseaseChip,
                          { backgroundColor: prediction.disease_info?.color + '20' || theme.colors.primary + '20' }
                        ]}
                      >
                        {prediction.class}
                      </Chip>
                      <Text style={styles.confidenceText}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </Text>
                    </View>
                    
                    {prediction.disease_info && (
                      <View style={styles.diseaseInfo}>
                        <Text style={styles.diseaseDescription}>
                          {prediction.disease_info.description}
                        </Text>
                        
                        {prediction.disease_info.symptoms.length > 0 && (
                          <View style={styles.symptomsContainer}>
                            <Text style={styles.symptomsTitle}>Symptoms:</Text>
                            {prediction.disease_info.symptoms.map((symptom, idx) => (
                              <Text key={idx} style={styles.symptomItem}>• {symptom}</Text>
                            ))}
                          </View>
                        )}
                        
                        <View style={styles.treatmentContainer}>
                          <Text style={styles.treatmentTitle}>Treatment:</Text>
                          <Text style={styles.treatmentText}>
                            {prediction.disease_info.treatment}
                          </Text>
                        </View>
                      </View>
                    )}
                    
                    {index < result.detailed_predictions.length - 1 && (
                      <Divider style={styles.predictionDivider} />
                    )}
                  </View>
                ))}
              </Card.Content>
            </Card>
          )}

          {/* Recommendations */}
          {result.recommendations.length > 0 && (
            <Card style={styles.recommendationsCard}>
              <Card.Content>
                <Title>Recommendations</Title>
                {result.recommendations.map((recommendation, index) => (
                  <Text key={index} style={styles.recommendationItem}>
                    • {recommendation}
                  </Text>
                ))}
              </Card.Content>
            </Card>
          )}

          {/* Action Buttons */}
          <View style={styles.actionButtons}>
            <Button
              mode="outlined"
              onPress={resetDetection}
              style={styles.actionButton}
            >
              Analyze Another
            </Button>
            <Button
              mode="contained"
              onPress={() => navigation.navigate('History')}
              style={styles.actionButton}
            >
              View History
            </Button>
          </View>
        </Animatable.View>
      </ScrollView>
    );
  };

  const renderMainInterface = (): JSX.Element => (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Animatable.View animation="fadeInDown" duration={800}>
          <Card style={styles.instructionCard}>
            <Card.Content>
              <Title>📷 Disease Detection</Title>
              <Paragraph>
                Take a clear photo of a maize leaf or upload an image from your gallery. 
                Ensure good lighting and include the entire leaf for best results.
              </Paragraph>
            </Card.Content>
          </Card>
        </Animatable.View>

        <Animatable.View animation="fadeInUp" delay={200} duration={800}>
          <View style={styles.buttonContainer}>
            <Button
              mode="contained"
              onPress={() => setShowCamera(true)}
              style={styles.primaryButton}
              contentStyle={styles.buttonContent}
              icon="camera"
            >
              Take Photo
            </Button>
            
            <Button
              mode="outlined"
              onPress={pickImage}
              style={styles.secondaryButton}
              contentStyle={styles.buttonContent}
              icon="image"
            >
              Choose from Gallery
            </Button>
          </View>
        </Animatable.View>

        {detection.isDetecting && (
          <Animatable.View animation="fadeIn" duration={500}>
            <Surface style={styles.loadingCard} elevation={2}>
              <ActivityIndicator size="large" color={theme.colors.primary} />
              <Text style={styles.loadingText}>Analyzing image...</Text>
              <Text style={styles.loadingSubtext}>
                Our AI is examining the leaf for diseases
              </Text>
            </Surface>
          </Animatable.View>
        )}
      </ScrollView>
    </View>
  );

  if (hasPermission === null) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
      </View>
    );
  }

  if (hasPermission === false) {
    return renderPermissionRequest();
  }

  if (showCamera) {
    return renderCameraView();
  }

  if (detection.result) {
    return renderDetectionResult();
  }

  return renderMainInterface();
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  scrollContent: {
    padding: 16,
  },
  instructionCard: {
    marginBottom: 24,
    borderRadius: 12,
  },
  buttonContainer: {
    gap: 16,
    marginBottom: 24,
  },
  primaryButton: {
    borderRadius: 25,
  },
  secondaryButton: {
    borderRadius: 25,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  loadingCard: {
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  loadingSubtext: {
    fontSize: 14,
    opacity: 0.7,
    textAlign: 'center',
  },
  permissionTitle: {
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 8,
  },
  permissionText: {
    textAlign: 'center',
    marginBottom: 24,
    opacity: 0.7,
  },
  permissionButton: {
    borderRadius: 25,
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
    justifyContent: 'space-between',
  },
  cameraHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 50,
  },
  closeButton: {
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  flipButton: {
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  cameraFooter: {
    padding: 20,
    paddingBottom: 50,
  },
  captureContainer: {
    alignItems: 'center',
  },
  captureButton: {
    borderRadius: 35,
    backgroundColor: theme.colors.primary,
  },
  captureButtonContent: {
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  resultContainer: {
    flex: 1,
    backgroundColor: theme.colors.background,
    padding: 16,
  },
  imageCard: {
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  resultImage: {
    width: '100%',
    height: 250,
    resizeMode: 'cover',
  },
  statusCard: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusText: {
    marginLeft: 16,
    flex: 1,
  },
  statusTitle: {
    fontSize: 18,
    marginBottom: 4,
  },
  statusSubtitle: {
    fontSize: 14,
    opacity: 0.7,
  },
  predictionsCard: {
    marginBottom: 16,
    borderRadius: 12,
  },
  predictionItem: {
    marginVertical: 8,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  diseaseChip: {
    alignSelf: 'flex-start',
  },
  confidenceText: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
  },
  diseaseInfo: {
    marginTop: 8,
  },
  diseaseDescription: {
    fontSize: 14,
    marginBottom: 12,
    fontStyle: 'italic',
  },
  symptomsContainer: {
    marginBottom: 12,
  },
  symptomsTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  symptomItem: {
    fontSize: 13,
    marginLeft: 8,
    marginBottom: 2,
  },
  treatmentContainer: {
    marginTop: 8,
  },
  treatmentTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  treatmentText: {
    fontSize: 13,
    lineHeight: 18,
  },
  predictionDivider: {
    marginTop: 16,
  },
  recommendationsCard: {
    marginBottom: 16,
    borderRadius: 12,
  },
  recommendationItem: {
    fontSize: 14,
    marginBottom: 8,
    lineHeight: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  actionButton: {
    flex: 1,
    borderRadius: 25,
  },
});

export default CameraScreen;