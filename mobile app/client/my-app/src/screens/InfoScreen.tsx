import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  Image,
} from 'react-native';
import {
  Text,
  Card,
  Searchbar,
  Chip,
  Surface,
  Button,
  Divider,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';

import { theme } from '../theme/theme';

const { width } = Dimensions.get('window');

interface DiseaseInfo {
  id: string;
  name: string;
  scientificName: string;
  description: string;
  symptoms: string[];
  causes: string[];
  treatment: string[];
  prevention: string[];
  severity: 'Low' | 'Medium' | 'High';
  prevalence: string;
  image: string;
  color: string;
}

const InfoScreen: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

  const diseaseData: DiseaseInfo[] = [
    {
      id: '1',
      name: 'Healthy Leaf',
      scientificName: 'Zea mays (Healthy)',
      description: 'A healthy maize leaf shows vibrant green coloration with no visible disease symptoms. This indicates optimal plant health and proper growing conditions.',
      symptoms: [
        'Vibrant green color',
        'No spots or lesions',
        'Normal leaf texture',
        'Proper leaf structure',
        'No yellowing or browning'
      ],
      causes: [
        'Optimal growing conditions',
        'Proper nutrition',
        'Adequate water supply',
        'Good soil health',
        'Disease-free environment'
      ],
      treatment: [
        'Continue current care practices',
        'Maintain regular monitoring',
        'Ensure consistent watering',
        'Apply balanced fertilizers as needed'
      ],
      prevention: [
        'Regular field inspection',
        'Proper plant spacing',
        'Balanced nutrition program',
        'Good drainage management',
        'Crop rotation practices'
      ],
      severity: 'Low',
      prevalence: 'Desired state',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      color: '#28a745'
    },
    {
      id: '2',
      name: 'Grey Leaf Spot',
      scientificName: 'Cercospora zeae-maydis',
      description: 'Grey leaf spot is a fungal disease that causes rectangular lesions on maize leaves. It thrives in warm, humid conditions and can significantly reduce yield if left untreated.',
      symptoms: [
        'Rectangular grey spots on leaves',
        'Lesions with yellow halos',
        'Spots may merge to form larger areas',
        'Premature leaf senescence',
        'Reduced photosynthetic area'
      ],
      causes: [
        'High humidity (>90%)',
        'Warm temperatures (22-30°C)',
        'Poor air circulation',
        'Dense plant canopy',
        'Infected crop residue'
      ],
      treatment: [
        'Apply fungicides (strobilurins, triazoles)',
        'Improve air circulation',
        'Remove infected plant debris',
        'Reduce plant density if possible',
        'Apply foliar fungicides preventively'
      ],
      prevention: [
        'Use resistant varieties',
        'Crop rotation with non-host plants',
        'Proper field sanitation',
        'Avoid overhead irrigation',
        'Maintain proper plant spacing'
      ],
      severity: 'Medium',
      prevalence: 'Common in humid regions',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      color: '#ffc107'
    },
    {
      id: '3',
      name: 'Northern Leaf Blight',
      scientificName: 'Exserohilum turcicum',
      description: 'Northern leaf blight is a serious fungal disease that causes large, cigar-shaped lesions on maize leaves. It can cause significant yield losses in susceptible varieties.',
      symptoms: [
        'Large, cigar-shaped lesions',
        'Greyish-green to tan colored spots',
        'Lesions may be 2.5-15 cm long',
        'Dark sporulation in lesion centers',
        'Leaves may die prematurely'
      ],
      causes: [
        'Cool, moist weather conditions',
        'High relative humidity',
        'Temperature range 18-27°C',
        'Infected seed or crop residue',
        'Wind-dispersed spores'
      ],
      treatment: [
        'Apply copper-based fungicides',
        'Use systemic fungicides (triazoles)',
        'Improve field drainage',
        'Remove infected plant material',
        'Apply fungicides at first symptom appearance'
      ],
      prevention: [
        'Plant resistant hybrids',
        'Crop rotation (2-3 years)',
        'Deep tillage to bury residue',
        'Balanced fertilization',
        'Avoid dense planting'
      ],
      severity: 'High',
      prevalence: 'Widespread in cooler climates',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      color: '#dc3545'
    },
    {
      id: '4',
      name: 'Maize Streak Virus',
      scientificName: 'Maize streak virus (MSV)',
      description: 'Maize Streak Virus is transmitted by leafhoppers and causes characteristic streaking patterns on leaves. It can cause severe stunting and yield reduction.',
      symptoms: [
        'Yellow streaks parallel to leaf veins',
        'Mosaic patterns on leaves',
        'Stunted plant growth',
        'Reduced ear size and grain fill',
        'Chlorotic streaking'
      ],
      causes: [
        'Leafhopper vector transmission',
        'Infected plant material',
        'Continuous maize cropping',
        'High leafhopper populations',
        'Favorable weather for vectors'
      ],
      treatment: [
        'Control leafhopper vectors',
        'Remove infected plants',
        'Apply insecticides for vector control',
        'Use virus-free planting material',
        'No direct chemical treatment for virus'
      ],
      prevention: [
        'Use resistant varieties',
        'Control leafhopper populations',
        'Early planting to avoid peak vector activity',
        'Remove volunteer maize plants',
        'Crop rotation with non-host crops'
      ],
      severity: 'High',
      prevalence: 'Common in sub-Saharan Africa',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      color: '#6f42c1'
    }
  ];

  const filteredDiseases = diseaseData.filter(disease => {
    const matchesSearch = disease.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         disease.scientificName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         disease.description.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesCategory = selectedCategory === 'All' || 
                           (selectedCategory === 'Healthy' && disease.name === 'Healthy Leaf') ||
                           (selectedCategory === 'Fungal' && disease.id !== '1' && disease.id !== '4') ||
                           (selectedCategory === 'Viral' && disease.id === '4') ||
                           (selectedCategory === disease.severity);
    
    return matchesSearch && matchesCategory;
  });

  const categories = ['All', 'Healthy', 'Fungal', 'Viral', 'Low', 'Medium', 'High'];

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'alert-circle';
      case 'medium': return 'warning';
      case 'low': return 'checkmark-circle';
      default: return 'information-circle';
    }
  };

  const toggleCardExpansion = (id: string) => {
    setExpandedCard(expandedCard === id ? null : id);
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={[theme.colors.primary, '#20c997']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Disease Information</Text>
          <Text style={styles.headerSubtitle}>
            Learn about maize diseases and treatments
          </Text>
        </View>
      </LinearGradient>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search diseases, symptoms, treatments..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchBar}
          iconColor={theme.colors.primary}
        />
      </View>

      {/* Category Filters */}
      <View style={styles.categoriesContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {categories.map((category) => (
            <Chip
              key={category}
              selected={selectedCategory === category}
              onPress={() => setSelectedCategory(category)}
              style={[
                styles.categoryChip,
                selectedCategory === category && styles.selectedCategoryChip
              ]}
              textStyle={[
                styles.categoryText,
                selectedCategory === category && styles.selectedCategoryText
              ]}
            >
              {category}
            </Chip>
          ))}
        </ScrollView>
      </View>

      {/* Disease Cards */}
      <ScrollView
        style={styles.contentContainer}
        showsVerticalScrollIndicator={false}
      >
        {filteredDiseases.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Ionicons name="search" size={64} color="#ccc" />
            <Text style={styles.emptyText}>No diseases found</Text>
            <Text style={styles.emptySubtext}>
              Try adjusting your search terms or filters
            </Text>
          </View>
        ) : (
          filteredDiseases.map((disease, index) => (
            <Animatable.View
              key={disease.id}
              animation="fadeInUp"
              delay={index * 100}
              style={styles.diseaseCard}
            >
              <Card style={styles.card}>
                <Card.Content style={styles.cardContent}>
                  {/* Disease Header */}
                  <View style={styles.diseaseHeader}>
                    <View style={styles.diseaseImageContainer}>
                      <Image
                        source={{ uri: disease.image }}
                        style={styles.diseaseImage}
                      />
                      <View
                        style={[
                          styles.severityBadge,
                          { backgroundColor: disease.color }
                        ]}
                      >
                        <Ionicons
                          name={getSeverityIcon(disease.severity) as any}
                          size={16}
                          color="white"
                        />
                      </View>
                    </View>
                    
                    <View style={styles.diseaseInfo}>
                      <Text style={styles.diseaseName}>{disease.name}</Text>
                      <Text style={styles.scientificName}>{disease.scientificName}</Text>
                      <View style={styles.diseaseMetrics}>
                        <Chip
                          style={[styles.severityChip, { backgroundColor: disease.color }]}
                          textStyle={styles.severityText}
                        >
                          {disease.severity} Risk
                        </Chip>
                        <Text style={styles.prevalenceText}>{disease.prevalence}</Text>
                      </View>
                    </View>
                  </View>

                  {/* Description */}
                  <Text style={styles.description}>{disease.description}</Text>

                  {/* Expand/Collapse Button */}
                  <Button
                    mode="text"
                    onPress={() => toggleCardExpansion(disease.id)}
                    style={styles.expandButton}
                    textColor={theme.colors.primary}
                  >
                    {expandedCard === disease.id ? 'Show Less' : 'Learn More'}
                    <Ionicons
                      name={expandedCard === disease.id ? 'chevron-up' : 'chevron-down'}
                      size={16}
                      color={theme.colors.primary}
                    />
                  </Button>

                  {/* Expanded Content */}
                  {expandedCard === disease.id && (
                    <Animatable.View animation="fadeInDown" duration={300}>
                      <Divider style={styles.divider} />
                      
                      {/* Symptoms */}
                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>
                          <Ionicons name="eye" size={16} color={theme.colors.primary} />
                          {' '}Symptoms
                        </Text>
                        {disease.symptoms.map((symptom, idx) => (
                          <Text key={idx} style={styles.listItem}>• {symptom}</Text>
                        ))}
                      </View>

                      {/* Causes */}
                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>
                          <Ionicons name="help-circle" size={16} color={theme.colors.warning} />
                          {' '}Causes
                        </Text>
                        {disease.causes.map((cause, idx) => (
                          <Text key={idx} style={styles.listItem}>• {cause}</Text>
                        ))}
                      </View>

                      {/* Treatment */}
                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>
                          <Ionicons name="medical" size={16} color={theme.colors.error} />
                          {' '}Treatment
                        </Text>
                        {disease.treatment.map((treatment, idx) => (
                          <Text key={idx} style={styles.listItem}>• {treatment}</Text>
                        ))}
                      </View>

                      {/* Prevention */}
                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>
                          <Ionicons name="shield-checkmark" size={16} color={theme.colors.success} />
                          {' '}Prevention
                        </Text>
                        {disease.prevention.map((prevention, idx) => (
                          <Text key={idx} style={styles.listItem}>• {prevention}</Text>
                        ))}
                      </View>
                    </Animatable.View>
                  )}
                </Card.Content>
              </Card>
            </Animatable.View>
          ))
        )}
        
        <View style={styles.bottomSpacing} />
      </ScrollView>

      {/* Quick Tips */}
      <Surface style={styles.tipsContainer} elevation={4}>
        <Text style={styles.tipsTitle}>💡 Quick Tips</Text>
        <Text style={styles.tipsText}>
          Early detection is key! Regular monitoring and proper identification can prevent major crop losses.
        </Text>
      </Surface>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    marginTop: 4,
  },
  searchContainer: {
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  searchBar: {
    backgroundColor: 'white',
  },
  categoriesContainer: {
    paddingLeft: 20,
    paddingBottom: 15,
  },
  categoryChip: {
    marginRight: 10,
    backgroundColor: 'white',
  },
  selectedCategoryChip: {
    backgroundColor: theme.colors.primary,
  },
  categoryText: {
    color: '#666',
  },
  selectedCategoryText: {
    color: 'white',
  },
  contentContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    marginTop: 16,
    fontWeight: '600',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  diseaseCard: {
    marginBottom: 16,
  },
  card: {
    borderRadius: 12,
    backgroundColor: 'white',
  },
  cardContent: {
    padding: 16,
  },
  diseaseHeader: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  diseaseImageContainer: {
    position: 'relative',
    marginRight: 15,
  },
  diseaseImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  severityBadge: {
    position: 'absolute',
    top: -5,
    right: -5,
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  diseaseInfo: {
    flex: 1,
  },
  diseaseName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  scientificName: {
    fontSize: 14,
    fontStyle: 'italic',
    color: '#666',
    marginBottom: 8,
  },
  diseaseMetrics: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  severityChip: {
    borderRadius: 12,
    height: 24,
  },
  severityText: {
    color: 'white',
    fontSize: 10,
    fontWeight: '600',
  },
  prevalenceText: {
    fontSize: 12,
    color: '#666',
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 12,
  },
  expandButton: {
    alignSelf: 'flex-start',
    marginTop: 8,
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
    color: '#333',
    marginBottom: 8,
  },
  listItem: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 4,
    paddingLeft: 8,
  },
  tipsContainer: {
    margin: 20,
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#e8f5e8',
  },
  tipsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
    marginBottom: 8,
  },
  tipsText: {
    fontSize: 14,
    color: '#2d5a2d',
    lineHeight: 18,
  },
  bottomSpacing: {
    height: 20,
  },
});

export default InfoScreen;