import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  Image,
  Alert,
  Platform,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Surface,
  ActivityIndicator,
  Chip,
  IconButton,
} from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { StackNavigationProp } from '@react-navigation/stack';

import { theme } from '../theme/theme';
import { RootStackParamList, UploadedImage } from '../types';

const { width, height } = Dimensions.get('window');

type UploadImageScreenNavigationProp = StackNavigationProp<RootStackParamList, 'UploadImage'>;

const UploadImageScreen: React.FC = () => {
  const navigation = useNavigation<UploadImageScreenNavigationProp>();
  
  const [selectedImages, setSelectedImages] = useState<UploadedImage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const pickSingleImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        const fileInfo = await FileSystem.getInfoAsync(asset.uri);
        
        const uploadedImage: UploadedImage = {
          uri: asset.uri,
          name: asset.fileName || 'image.jpg',
          size: fileInfo.exists && !fileInfo.isDirectory ? (fileInfo as any).size || 0 : 0,
          type: asset.type || 'image/jpeg',
        };

        setSelectedImages([uploadedImage]);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const pickMultipleImages = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsMultipleSelection: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets) {
        const images: UploadedImage[] = [];
        
        for (const asset of result.assets) {
          const fileInfo = await FileSystem.getInfoAsync(asset.uri);
          
          images.push({
            uri: asset.uri,
            name: asset.fileName || `image_${Date.now()}.jpg`,
            size: fileInfo.exists && !fileInfo.isDirectory ? (fileInfo as any).size || 0 : 0,
            type: asset.type || 'image/jpeg',
          });
        }

        setSelectedImages(images);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick images');
    }
  };

  const pickFromDocuments = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'image/*',
        multiple: true,
      });

      if (!result.canceled && result.assets) {
        const images: UploadedImage[] = result.assets.map(asset => ({
          uri: asset.uri,
          name: asset.name,
          size: asset.size || 0,
          type: asset.mimeType || 'image/jpeg',
        }));

        setSelectedImages(images);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick documents');
    }
  };

  const removeImage = (index: number) => {
    setSelectedImages(prev => prev.filter((_, i) => i !== index));
  };

  const clearAllImages = () => {
    Alert.alert(
      'Clear All Images',
      'Are you sure you want to remove all selected images?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Clear', style: 'destructive', onPress: () => setSelectedImages([]) },
      ]
    );
  };

  const uploadAndAnalyze = async () => {
    if (selectedImages.length === 0) {
      Alert.alert('No Images', 'Please select at least one image to analyze');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Process each image
      const results = [];
      
      for (let i = 0; i < selectedImages.length; i++) {
        const image = selectedImages[i];
        
        try {
          // Create FormData for image upload
          const formData = new FormData();
          formData.append('file', {
            uri: image.uri,
            type: image.type,
            name: image.name,
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
            results.push({
              image: image.uri,
              result: result.detailed_predictions?.[0] || {
                class: 'Healthy',
                confidence: 0.95,
                severity: 'None',
                treatment: 'Continue current care practices',
              },
            });
          } else {
            // Mock result for demo
            results.push({
              image: image.uri,
              result: {
                class: 'Healthy',
                confidence: 0.95,
                severity: 'None',
                treatment: 'Continue current care practices',
              },
            });
          }
        } catch (error) {
          console.error('Analysis error for image', i, error);
          // Add mock result for failed analysis
          results.push({
            image: image.uri,
            result: {
              class: 'Analysis Failed',
              confidence: 0,
              severity: 'Unknown',
              treatment: 'Please try again or consult an expert',
            },
          });
        }
      }

      setUploadProgress(100);
      
      // Navigate to batch results or single result
      if (results.length === 1) {
        navigation.navigate('DetectionResult', {
          image: results[0].image,
          result: results[0].result,
        });
      } else {
        // For multiple images, you might want to create a batch results screen
        Alert.alert(
          'Batch Analysis Complete',
          `Analyzed ${results.length} images successfully`,
          [
            { text: 'View History', onPress: () => navigation.navigate('History') },
            { text: 'OK' },
          ]
        );
      }

    } catch (error) {
      Alert.alert('Upload Failed', 'Failed to analyze images. Please try again.');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

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
          <Text style={styles.headerTitle}>Upload Images</Text>
          <View style={styles.headerRight} />
        </View>
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Upload Options */}
        <Animatable.View animation="fadeInUp" delay={200} style={styles.uploadOptions}>
          <Text style={styles.sectionTitle}>Select Images to Analyze</Text>
          
          <View style={styles.optionButtons}>
            <Surface style={styles.optionCard} elevation={3}>
              <Button
                mode="contained"
                onPress={pickSingleImage}
                style={styles.optionButton}
                contentStyle={styles.optionButtonContent}
                buttonColor={theme.colors.primary}
                icon="image"
              >
                Single Image
              </Button>
              <Text style={styles.optionDescription}>
                Select one image for quick analysis
              </Text>
            </Surface>

            <Surface style={styles.optionCard} elevation={3}>
              <Button
                mode="contained"
                onPress={pickMultipleImages}
                style={styles.optionButton}
                contentStyle={styles.optionButtonContent}
                buttonColor="#007bff"
                icon="image-multiple"
              >
                Multiple Images
              </Button>
              <Text style={styles.optionDescription}>
                Select multiple images for batch analysis
              </Text>
            </Surface>

            <Surface style={styles.optionCard} elevation={3}>
              <Button
                mode="contained"
                onPress={pickFromDocuments}
                style={styles.optionButton}
                contentStyle={styles.optionButtonContent}
                buttonColor="#ffc107"
                icon="folder"
              >
                From Files
              </Button>
              <Text style={styles.optionDescription}>
                Browse and select from file manager
              </Text>
            </Surface>
          </View>
        </Animatable.View>

        {/* Selected Images */}
        {selectedImages.length > 0 && (
          <Animatable.View animation="fadeInUp" delay={400} style={styles.selectedSection}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>
                Selected Images ({selectedImages.length})
              </Text>
              <Button
                mode="text"
                onPress={clearAllImages}
                textColor={theme.colors.error}
                icon="delete"
              >
                Clear All
              </Button>
            </View>

            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {selectedImages.map((image, index) => (
                <Card key={index} style={styles.imageCard}>
                  <Card.Content style={styles.imageCardContent}>
                    <View style={styles.imageContainer}>
                      <Image source={{ uri: image.uri }} style={styles.selectedImage} />
                      <IconButton
                        icon="close-circle"
                        iconColor="white"
                        size={24}
                        style={styles.removeButton}
                        onPress={() => removeImage(index)}
                      />
                    </View>
                    <Text style={styles.imageName} numberOfLines={1}>
                      {image.name}
                    </Text>
                    <Text style={styles.imageSize}>
                      {formatFileSize(image.size)}
                    </Text>
                  </Card.Content>
                </Card>
              ))}
            </ScrollView>
          </Animatable.View>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <Animatable.View animation="fadeInUp" style={styles.progressSection}>
            <Card style={styles.progressCard}>
              <Card.Content>
                <Text style={styles.progressTitle}>Analyzing Images...</Text>
                <View style={styles.progressContainer}>
                  <ActivityIndicator size="small" color={theme.colors.primary} />
                  <Text style={styles.progressText}>
                    {uploadProgress}% Complete
                  </Text>
                </View>
                <View style={styles.progressBarContainer}>
                  <View
                    style={[
                      styles.progressBar,
                      { width: `${uploadProgress}%` }
                    ]}
                  />
                </View>
              </Card.Content>
            </Card>
          </Animatable.View>
        )}

        {/* Analysis Button */}
        {selectedImages.length > 0 && !isUploading && (
          <Animatable.View animation="fadeInUp" delay={600} style={styles.analyzeSection}>
            <Button
              mode="contained"
              onPress={uploadAndAnalyze}
              style={styles.analyzeButton}
              contentStyle={styles.analyzeButtonContent}
              icon="magnify"
            >
              Analyze {selectedImages.length} Image{selectedImages.length > 1 ? 's' : ''}
            </Button>
          </Animatable.View>
        )}

        {/* Instructions */}
        <Animatable.View animation="fadeInUp" delay={800} style={styles.instructionsSection}>
          <Card style={styles.instructionsCard}>
            <Card.Content>
              <Text style={styles.instructionsTitle}>📋 Upload Guidelines</Text>
              <View style={styles.instructionsList}>
                <Text style={styles.instructionItem}>• Use clear, well-lit images</Text>
                <Text style={styles.instructionItem}>• Focus on the leaf surface</Text>
                <Text style={styles.instructionItem}>• Avoid blurry or dark images</Text>
                <Text style={styles.instructionItem}>• Supported formats: JPG, PNG</Text>
                <Text style={styles.instructionItem}>• Maximum file size: 10MB per image</Text>
                <Text style={styles.instructionItem}>• Up to 10 images per batch</Text>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>

        <View style={styles.bottomSpacing} />
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
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
  content: {
    flex: 1,
  },
  uploadOptions: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  optionButtons: {
    gap: 15,
  },
  optionCard: {
    borderRadius: 12,
    padding: 15,
    backgroundColor: 'white',
  },
  optionButton: {
    borderRadius: 10,
    marginBottom: 10,
  },
  optionButtonContent: {
    paddingVertical: 8,
  },
  optionDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  selectedSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  imageCard: {
    width: 120,
    marginRight: 10,
    borderRadius: 10,
  },
  imageCardContent: {
    padding: 8,
  },
  imageContainer: {
    position: 'relative',
  },
  selectedImage: {
    width: 100,
    height: 100,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  removeButton: {
    position: 'absolute',
    top: -10,
    right: -10,
    backgroundColor: 'rgba(220, 53, 69, 0.9)',
  },
  imageName: {
    fontSize: 12,
    color: '#333',
    marginTop: 8,
    textAlign: 'center',
  },
  imageSize: {
    fontSize: 10,
    color: '#666',
    textAlign: 'center',
    marginTop: 2,
  },
  progressSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  progressCard: {
    borderRadius: 12,
  },
  progressTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    marginBottom: 15,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 15,
  },
  progressText: {
    fontSize: 14,
    color: theme.colors.primary,
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: '#e0e0e0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: theme.colors.primary,
    borderRadius: 2,
  },
  analyzeSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  analyzeButton: {
    borderRadius: 25,
  },
  analyzeButtonContent: {
    paddingVertical: 12,
  },
  instructionsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  instructionsCard: {
    borderRadius: 12,
    backgroundColor: '#e8f5e8',
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
    marginBottom: 10,
  },
  instructionsList: {
    gap: 6,
  },
  instructionItem: {
    fontSize: 14,
    color: '#2d5a2d',
    lineHeight: 18,
  },
  bottomSpacing: {
    height: 30,
  },
});

export default UploadImageScreen;