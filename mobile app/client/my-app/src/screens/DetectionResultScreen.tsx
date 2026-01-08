import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  Image,
  Share,
  Alert,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Chip,
  Surface,
  Divider,
  IconButton,
  ProgressBar,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import type { StackNavigationProp, StackScreenProps } from '@react-navigation/stack';

import { theme } from '../theme/theme';
import { RootStackParamList, DetectionResult } from '../types';

const { width } = Dimensions.get('window');

type DetectionResultScreenNavigationProp = StackNavigationProp<RootStackParamList, 'DetectionResult'>;
type DetectionResultScreenProps = StackScreenProps<RootStackParamList, 'DetectionResult'>;

const DetectionResultScreen: React.FC = () => {
  const navigation = useNavigation<DetectionResultScreenNavigationProp>();
  const route = useRoute<DetectionResultScreenProps['route']>();
  const { image, result, fromHistory } = route.params;
  
  const [showFullTreatment, setShowFullTreatment] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const diseaseInfo = {
    'Healthy': {
      description: 'Your maize leaf appears to be healthy with no visible disease symptoms.',
      symptoms: ['Vibrant green color', 'No spots or lesions', 'Normal texture'],
      prevention: ['Continue current care practices', 'Regular monitoring', 'Proper nutrition'],
      icon: 'checkmark-circle',
      color: '#28a745',
      riskLevel: 'No Risk'
    },
    'Grey Leaf Spots': {
      description: 'Grey leaf spot is a fungal disease caused by Cercospora zeae-maydis.',
      symptoms: ['Rectangular grey spots', 'Yellow halos around lesions', 'Reduced photosynthesis'],
      prevention: ['Use resistant varieties', 'Improve air circulation', 'Crop rotation'],
      icon: 'warning',
      color: '#ffc107',
      riskLevel: 'Medium Risk'
    },
    'Leaf Blight': {
      description: 'Northern leaf blight is a serious fungal disease that can cause significant yield loss.',
      symptoms: ['Large cigar-shaped lesions', 'Greyish-green to tan spots', 'Premature leaf death'],
      prevention: ['Plant resistant hybrids', 'Deep tillage', 'Balanced fertilization'],
      icon: 'alert-circle',
      color: '#dc3545',
      riskLevel: 'High Risk'
    },
    'MSV': {
      description: 'Maize Streak Virus is transmitted by leafhoppers and causes characteristic streaking.',
      symptoms: ['Yellow streaks parallel to veins', 'Stunted growth', 'Mosaic patterns'],
      prevention: ['Control leafhopper vectors', 'Use virus-free seeds', 'Early planting'],
      icon: 'bug',
      color: '#6f42c1',
      riskLevel: 'High Risk'
    }
  };

  const currentDiseaseInfo = diseaseInfo[result.class as keyof typeof diseaseInfo] || diseaseInfo['Healthy'];

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return '#28a745';
    if (confidence >= 70) return '#ffc107';
    return '#dc3545';
  };

  const handleSaveResult = async () => {
    setIsSaving(true);
    try {
      // Simulate saving to local storage or API
      await new Promise(resolve => setTimeout(resolve, 1500));
      Alert.alert('Success', 'Detection result saved to history');
    } catch (error) {
      Alert.alert('Error', 'Failed to save result');
    } finally {
      setIsSaving(false);
    }
  };

  const handleShareResult = async () => {
    try {
      const shareContent = {
        message: `Maize Disease Detection Result:
Disease: ${result.class}
Confidence: ${result.confidence.toFixed(1)}%
Severity: ${result.severity}
Treatment: ${result.treatment}

Detected using AI-powered Maize Disease Detection App`,
        title: 'Disease Detection Result'
      };
      
      await Share.share(shareContent);
    } catch (error) {
      Alert.alert('Error', 'Failed to share result');
    }
  };

  const handleNewScan = () => {
    navigation.navigate('Camera');
  };

  const handleViewHistory = () => {
    navigation.navigate('History');
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={[currentDiseaseInfo.color, currentDiseaseInfo.color + '80']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <IconButton
            icon="arrow-left"
            iconColor="white"
            size={24}
            onPress={() => navigation.goBack()}
          />
          <Text style={styles.headerTitle}>Detection Result</Text>
          <IconButton
            icon="share"
            iconColor="white"
            size={24}
            onPress={handleShareResult}
          />
        </View>
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Image Display */}
        <Animatable.View animation="fadeInUp" delay={200} style={styles.imageSection}>
          <Card style={styles.imageCard}>
            <Card.Content style={styles.imageContent}>
              <Image source={{ uri: image }} style={styles.leafImage} />
              <View style={styles.imageOverlay}>
                <Surface style={styles.confidenceBadge} elevation={4}>
                  <Text style={styles.confidenceText}>
                    {result.confidence.toFixed(1)}%
                  </Text>
                  <Text style={styles.confidenceLabel}>Confidence</Text>
                </Surface>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Result Summary */}
        <Animatable.View animation="fadeInUp" delay={400} style={styles.resultSection}>
          <Card style={styles.resultCard}>
            <Card.Content style={styles.resultContent}>
              <View style={styles.resultHeader}>
                <View style={styles.diseaseIcon}>
                  <Ionicons
                    name={currentDiseaseInfo.icon as any}
                    size={32}
                    color={currentDiseaseInfo.color}
                  />
                </View>
                <View style={styles.resultInfo}>
                  <Text style={styles.diseaseName}>{result.class}</Text>
                  <Text style={styles.diseaseDescription}>
                    {currentDiseaseInfo.description}
                  </Text>
                </View>
              </View>

              <View style={styles.metricsRow}>
                <Surface style={styles.metricCard} elevation={2}>
                  <Text style={styles.metricValue}>{result.confidence.toFixed(1)}%</Text>
                  <Text style={styles.metricLabel}>Confidence</Text>
                  <ProgressBar
                    progress={result.confidence / 100}
                    color={getConfidenceColor(result.confidence)}
                    style={styles.progressBar}
                  />
                </Surface>

                <Surface style={styles.metricCard} elevation={2}>
                  <Chip
                    style={[styles.riskChip, { backgroundColor: currentDiseaseInfo.color }]}
                    textStyle={styles.riskText}
                  >
                    {currentDiseaseInfo.riskLevel}
                  </Chip>
                  <Text style={styles.metricLabel}>Risk Level</Text>
                </Surface>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Symptoms */}
        <Animatable.View animation="fadeInUp" delay={600} style={styles.section}>
          <Card style={styles.sectionCard}>
            <Card.Content>
              <Text style={styles.sectionTitle}>
                <Ionicons name="eye" size={18} color={theme.colors.primary} />
                {' '}Symptoms to Look For
              </Text>
              {currentDiseaseInfo.symptoms.map((symptom, index) => (
                <View key={index} style={styles.symptomItem}>
                  <Ionicons name="ellipse" size={6} color={currentDiseaseInfo.color} />
                  <Text style={styles.symptomText}>{symptom}</Text>
                </View>
              ))}
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Treatment */}
        <Animatable.View animation="fadeInUp" delay={800} style={styles.section}>
          <Card style={styles.sectionCard}>
            <Card.Content>
              <Text style={styles.sectionTitle}>
                <Ionicons name="medical" size={18} color={theme.colors.error} />
                {' '}Recommended Treatment
              </Text>
              <Text style={styles.treatmentText}>
                {showFullTreatment ? result.treatment : `${result.treatment.substring(0, 100)}...`}
              </Text>
              {result.treatment.length > 100 && (
                <Button
                  mode="text"
                  onPress={() => setShowFullTreatment(!showFullTreatment)}
                  textColor={theme.colors.primary}
                  style={styles.showMoreButton}
                >
                  {showFullTreatment ? 'Show Less' : 'Show More'}
                </Button>
              )}
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Prevention */}
        <Animatable.View animation="fadeInUp" delay={1000} style={styles.section}>
          <Card style={styles.sectionCard}>
            <Card.Content>
              <Text style={styles.sectionTitle}>
                <Ionicons name="shield-checkmark" size={18} color={theme.colors.success} />
                {' '}Prevention Tips
              </Text>
              {currentDiseaseInfo.prevention.map((tip, index) => (
                <View key={index} style={styles.preventionItem}>
                  <Ionicons name="checkmark-circle" size={16} color={theme.colors.success} />
                  <Text style={styles.preventionText}>{tip}</Text>
                </View>
              ))}
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Action Buttons */}
        <Animatable.View animation="fadeInUp" delay={1200} style={styles.actionsSection}>
          <View style={styles.actionButtons}>
            {!fromHistory && (
              <Button
                mode="outlined"
                onPress={handleSaveResult}
                loading={isSaving}
                disabled={isSaving}
                style={styles.actionButton}
                icon="content-save"
              >
                {isSaving ? 'Saving...' : 'Save Result'}
              </Button>
            )}
            
            <Button
              mode="contained"
              onPress={handleNewScan}
              style={styles.actionButton}
              icon="camera"
            >
              New Scan
            </Button>
          </View>

          <Button
            mode="text"
            onPress={handleViewHistory}
            textColor={theme.colors.primary}
            style={styles.historyButton}
            icon="history"
          >
            View Detection History
          </Button>
        </Animatable.View>

        {/* Additional Info */}
        <Animatable.View animation="fadeInUp" delay={1400} style={styles.infoSection}>
          <Surface style={styles.infoCard} elevation={2}>
            <Text style={styles.infoTitle}>ℹ️ Important Note</Text>
            <Text style={styles.infoText}>
              This AI-powered detection provides guidance based on visual analysis. 
              For severe cases or persistent problems, consult with a local agricultural expert 
              or extension officer for comprehensive treatment plans.
            </Text>
          </Surface>
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
    paddingTop: 50,
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
  content: {
    flex: 1,
  },
  imageSection: {
    padding: 20,
  },
  imageCard: {
    borderRadius: 15,
    overflow: 'hidden',
  },
  imageContent: {
    padding: 0,
    position: 'relative',
  },
  leafImage: {
    width: '100%',
    height: 250,
    resizeMode: 'cover',
  },
  imageOverlay: {
    position: 'absolute',
    top: 15,
    right: 15,
  },
  confidenceBadge: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    alignItems: 'center',
  },
  confidenceText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceLabel: {
    fontSize: 10,
    color: '#666',
    marginTop: 2,
  },
  resultSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  resultCard: {
    borderRadius: 15,
  },
  resultContent: {
    padding: 20,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 20,
  },
  diseaseIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(46, 139, 87, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  resultInfo: {
    flex: 1,
  },
  diseaseName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  diseaseDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  metricsRow: {
    flexDirection: 'row',
    gap: 15,
  },
  metricCard: {
    flex: 1,
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: 'white',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  progressBar: {
    width: '100%',
    height: 4,
    marginTop: 8,
    borderRadius: 2,
  },
  riskChip: {
    borderRadius: 15,
  },
  riskText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 15,
  },
  sectionCard: {
    borderRadius: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 15,
  },
  symptomItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    paddingLeft: 10,
  },
  symptomText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 10,
    flex: 1,
  },
  treatmentText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 22,
  },
  showMoreButton: {
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  preventionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
    paddingLeft: 5,
  },
  preventionText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 10,
    flex: 1,
    lineHeight: 20,
  },
  actionsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  actionButton: {
    flex: 1,
  },
  historyButton: {
    alignSelf: 'center',
  },
  infoSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  infoCard: {
    padding: 15,
    borderRadius: 12,
    backgroundColor: '#e3f2fd',
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1976d2',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 13,
    color: '#1565c0',
    lineHeight: 18,
  },
  bottomSpacing: {
    height: 30,
  },
});

export default DetectionResultScreen;