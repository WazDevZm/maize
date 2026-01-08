import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Text,
  Surface,
  Chip,
  ActivityIndicator,
  Divider,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

import { theme } from '@/theme/theme';
import { apiService } from '@/config/api';
import type { RootTabParamList, DiseaseInfo, DiseasesResponse } from '@/types';

const { width } = Dimensions.get('window');

type Props = BottomTabScreenProps<RootTabParamList, 'Info'>;

const InfoScreen: React.FC<Props> = ({ navigation }) => {
  const [diseases, setDiseases] = useState<Record<string, DiseaseInfo>>({});
  const [loading, setLoading] = useState(true);
  const [selectedDisease, setSelectedDisease] = useState<string | null>(null);

  useEffect(() => {
    loadDiseaseInfo();
  }, []);

  const loadDiseaseInfo = async (): Promise<void> => {
    try {
      setLoading(true);
      const response: DiseasesResponse = await apiService.getDiseases();
      setDiseases(response.diseases);
    } catch (error) {
      console.error('Error loading disease info:', error);
      // Fallback to local disease data
      setDiseases(fallbackDiseaseData);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case 'none': return theme.colors.success;
      case 'low': return theme.colors.info;
      case 'medium': return theme.colors.warning;
      case 'high': return theme.colors.error;
      default: return theme.colors.primary;
    }
  };

  const getSeverityIcon = (severity: string): keyof typeof Ionicons.glyphMap => {
    switch (severity.toLowerCase()) {
      case 'none': return 'checkmark-circle';
      case 'low': return 'information-circle';
      case 'medium': return 'warning';
      case 'high': return 'alert-circle';
      default: return 'help-circle';
    }
  };

  const renderDiseaseCard = (diseaseName: string, diseaseInfo: DiseaseInfo, index: number): JSX.Element => {
    const isSelected = selectedDisease === diseaseName;
    
    return (
      <Animatable.View
        key={diseaseName}
        animation="fadeInUp"
        delay={index * 150}
        duration={600}
      >
        <Card
          style={[
            styles.diseaseCard,
            isSelected && styles.selectedCard
          ]}
          onPress={() => setSelectedDisease(isSelected ? null : diseaseName)}
        >
          <Card.Content>
            <View style={styles.diseaseHeader}>
              <View style={styles.diseaseTitle}>
                <Title style={styles.diseaseName}>{diseaseName.replace(/_/g, ' ')}</Title>
                <View style={styles.severityContainer}>
                  <Ionicons
                    name={getSeverityIcon(diseaseInfo.severity)}
                    size={20}
                    color={getSeverityColor(diseaseInfo.severity)}
                  />
                  <Chip
                    style={[
                      styles.severityChip,
                      { backgroundColor: getSeverityColor(diseaseInfo.severity) + '20' }
                    ]}
                    textStyle={[
                      styles.severityText,
                      { color: getSeverityColor(diseaseInfo.severity) }
                    ]}
                  >
                    {diseaseInfo.severity} Severity
                  </Chip>
                </View>
              </View>
              <Ionicons
                name={isSelected ? "chevron-up" : "chevron-down"}
                size={24}
                color={theme.colors.primary}
              />
            </View>

            <Paragraph style={styles.diseaseDescription}>
              {diseaseInfo.description}
            </Paragraph>

            {isSelected && (
              <Animatable.View animation="fadeIn" duration={300}>
                <Divider style={styles.divider} />
                
                {/* Symptoms */}
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>🔍 Symptoms</Text>
                  {diseaseInfo.symptoms.map((symptom, idx) => (
                    <View key={idx} style={styles.listItem}>
                      <Text style={styles.bullet}>•</Text>
                      <Text style={styles.listText}>{symptom}</Text>
                    </View>
                  ))}
                </View>

                {/* Treatment */}
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>💊 Treatment</Text>
                  <Surface style={styles.treatmentCard} elevation={1}>
                    <Text style={styles.treatmentText}>{diseaseInfo.treatment}</Text>
                  </Surface>
                </View>

                {/* Prevention */}
                {diseaseInfo.prevention && diseaseInfo.prevention.length > 0 && (
                  <View style={styles.section}>
                    <Text style={styles.sectionTitle}>🛡️ Prevention</Text>
                    {diseaseInfo.prevention.map((prevention, idx) => (
                      <View key={idx} style={styles.listItem}>
                        <Text style={styles.bullet}>•</Text>
                        <Text style={styles.listText}>{prevention}</Text>
                      </View>
                    ))}
                  </View>
                )}
              </Animatable.View>
            )}
          </Card.Content>
        </Card>
      </Animatable.View>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading disease information...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <LinearGradient
        colors={[theme.colors.primary, '#1e6b47']}
        style={styles.header}
      >
        <Animatable.View animation="fadeInDown" duration={1000}>
          <Text style={styles.headerTitle}>🦠 Disease Information</Text>
          <Text style={styles.headerSubtitle}>
            Learn about maize diseases, symptoms, and treatments
          </Text>
        </Animatable.View>
      </LinearGradient>

      <View style={styles.content}>
        {/* Overview */}
        <Animatable.View animation="fadeInUp" delay={200}>
          <Surface style={styles.overviewCard} elevation={2}>
            <View style={styles.overviewContent}>
              <Ionicons name="information-circle" size={32} color={theme.colors.primary} />
              <View style={styles.overviewText}>
                <Title style={styles.overviewTitle}>Disease Detection System</Title>
                <Text style={styles.overviewDescription}>
                  Our AI system can detect {Object.keys(diseases).length} different conditions in maize leaves with 99.5% accuracy.
                  Tap on any disease below to learn more about symptoms and treatments.
                </Text>
              </View>
            </View>
          </Surface>
        </Animatable.View>

        {/* Disease Cards */}
        <View style={styles.diseasesContainer}>
          {Object.entries(diseases).map(([diseaseName, diseaseInfo], index) =>
            renderDiseaseCard(diseaseName, diseaseInfo, index)
          )}
        </View>

        {/* Tips Section */}
        <Animatable.View animation="fadeInUp" delay={800}>
          <Card style={styles.tipsCard}>
            <Card.Content>
              <Title style={styles.tipsTitle}>💡 Detection Tips</Title>
              <View style={styles.tipsList}>
                <View style={styles.tipItem}>
                  <Ionicons name="camera" size={20} color={theme.colors.primary} />
                  <Text style={styles.tipText}>
                    Take clear, well-lit photos of maize leaves
                  </Text>
                </View>
                <View style={styles.tipItem}>
                  <Ionicons name="leaf" size={20} color={theme.colors.primary} />
                  <Text style={styles.tipText}>
                    Include the entire leaf in the frame
                  </Text>
                </View>
                <View style={styles.tipItem}>
                  <Ionicons name="sunny" size={20} color={theme.colors.primary} />
                  <Text style={styles.tipText}>
                    Ensure good lighting conditions
                  </Text>
                </View>
                <View style={styles.tipItem}>
                  <Ionicons name="hand-left" size={20} color={theme.colors.primary} />
                  <Text style={styles.tipText}>
                    Hold the camera steady to avoid blur
                  </Text>
                </View>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* About Section */}
        <Animatable.View animation="fadeInUp" delay={1000}>
          <Card style={styles.aboutCard}>
            <Card.Content>
              <Title style={styles.aboutTitle}>🤖 About the AI Model</Title>
              <Text style={styles.aboutText}>
                This application uses YOLOv8 (You Only Look Once version 8), a state-of-the-art 
                computer vision model trained specifically on maize disease detection. The model 
                has been trained on thousands of maize leaf images and achieves 99.5% accuracy 
                in identifying diseases.
              </Text>
              
              <View style={styles.modelStats}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>99.5%</Text>
                  <Text style={styles.statLabel}>Accuracy</Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>4</Text>
                  <Text style={styles.statLabel}>Disease Types</Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>YOLOv8</Text>
                  <Text style={styles.statLabel}>AI Model</Text>
                </View>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>
      </View>
    </ScrollView>
  );
};

// Fallback disease data in case API fails
const fallbackDiseaseData: Record<string, DiseaseInfo> = {
  "Health": {
    description: "Healthy maize leaf with no visible disease symptoms",
    symptoms: ["Green color", "No spots", "Normal texture", "No lesions"],
    treatment: "Continue current care practices",
    severity: "None",
    color: "#28a745",
    prevention: [
      "Maintain proper spacing between plants",
      "Ensure adequate nutrition",
      "Regular monitoring"
    ]
  },
  "Grey_Leaf_Spots": {
    description: "Grey leaf spot disease caused by Cercospora zeae-maydis",
    symptoms: ["Grey spots", "Lesions on leaves", "Yellowing", "Reduced photosynthesis"],
    treatment: "Apply fungicides, improve air circulation, remove infected leaves",
    severity: "Medium",
    color: "#ffc107",
    prevention: [
      "Crop rotation with non-host plants",
      "Resistant varieties",
      "Proper field sanitation"
    ]
  },
  "Leaf_Blight": {
    description: "Leaf blight disease caused by Helminthosporium maydis",
    symptoms: ["Brown lesions", "Leaf wilting", "Yellow halos", "Premature leaf death"],
    treatment: "Apply copper-based fungicides, improve drainage, crop rotation",
    severity: "High",
    color: "#dc3545",
    prevention: [
      "Use resistant varieties",
      "Improve field drainage",
      "Balanced fertilization"
    ]
  },
  "MSV": {
    description: "Maize Streak Virus transmitted by leafhoppers",
    symptoms: ["Yellow streaks", "Stunted growth", "Mosaic patterns", "Reduced yield"],
    treatment: "Control leafhoppers, use resistant varieties, remove infected plants",
    severity: "High",
    color: "#dc3545",
    prevention: [
      "Control leafhopper vectors",
      "Use virus-free seeds",
      "Early planting"
    ]
  }
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
  loadingText: {
    fontSize: 16,
    marginTop: 16,
    color: theme.colors.onBackground,
  },
  header: {
    padding: 24,
    paddingTop: 40,
    paddingBottom: 60,
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'white',
    textAlign: 'center',
    opacity: 0.9,
  },
  content: {
    padding: 16,
    marginTop: -30,
  },
  overviewCard: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 24,
  },
  overviewContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  overviewText: {
    marginLeft: 16,
    flex: 1,
  },
  overviewTitle: {
    fontSize: 18,
    marginBottom: 8,
  },
  overviewDescription: {
    fontSize: 14,
    lineHeight: 20,
    opacity: 0.8,
  },
  diseasesContainer: {
    marginBottom: 24,
  },
  diseaseCard: {
    marginBottom: 12,
    borderRadius: 12,
  },
  selectedCard: {
    borderColor: theme.colors.primary,
    borderWidth: 2,
  },
  diseaseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  diseaseTitle: {
    flex: 1,
  },
  diseaseName: {
    fontSize: 18,
    marginBottom: 8,
  },
  severityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  severityChip: {
    marginLeft: 8,
  },
  severityText: {
    fontSize: 12,
    fontWeight: '600',
  },
  diseaseDescription: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  divider: {
    marginVertical: 16,
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: theme.colors.primary,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 4,
  },
  bullet: {
    fontSize: 14,
    marginRight: 8,
    marginTop: 2,
    color: theme.colors.primary,
  },
  listText: {
    fontSize: 14,
    flex: 1,
    lineHeight: 18,
  },
  treatmentCard: {
    padding: 12,
    borderRadius: 8,
    backgroundColor: theme.colors.surface,
  },
  treatmentText: {
    fontSize: 14,
    lineHeight: 18,
  },
  tipsCard: {
    borderRadius: 12,
    marginBottom: 24,
  },
  tipsTitle: {
    fontSize: 18,
    marginBottom: 16,
    color: theme.colors.primary,
  },
  tipsList: {
    gap: 12,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  tipText: {
    fontSize: 14,
    marginLeft: 12,
    flex: 1,
  },
  aboutCard: {
    borderRadius: 12,
    marginBottom: 24,
  },
  aboutTitle: {
    fontSize: 18,
    marginBottom: 12,
    color: theme.colors.primary,
  },
  aboutText: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 16,
  },
  modelStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: theme.colors.outline,
    borderTopStyle: 'solid',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  statLabel: {
    fontSize: 12,
    marginTop: 4,
    opacity: 0.7,
  },
});

export default InfoScreen;