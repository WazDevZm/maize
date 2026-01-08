import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  RefreshControl,
  Alert,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Surface,
  Chip,
  ProgressBar,
  Divider,
  IconButton,
  FAB,
  Menu,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { StackNavigationProp } from '@react-navigation/stack';

import { theme } from '../theme/theme';
import { RootStackParamList, User } from '../types';

const { width } = Dimensions.get('window');

interface Field {
  id: string;
  name: string;
  area: number; // in acres
  cropType: string;
  plantingDate: string;
  expectedHarvest: string;
  healthStatus: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  lastInspection: string;
  diseaseCount: number;
  yieldPrediction: number;
  location: {
    latitude: number;
    longitude: number;
  };
}

interface FarmTrackerScreenProps {
  user?: User | null;
}

type FarmTrackerScreenNavigationProp = StackNavigationProp<RootStackParamList, 'FarmTracker'>;

const FarmTrackerScreen: React.FC<FarmTrackerScreenProps> = ({ user }) => {
  const navigation = useNavigation<FarmTrackerScreenNavigationProp>();
  const [refreshing, setRefreshing] = useState(false);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [menuVisible, setMenuVisible] = useState(false);

  const [fields, setFields] = useState<Field[]>([
    {
      id: '1',
      name: 'North Field A',
      area: 25.5,
      cropType: 'Maize',
      plantingDate: '2024-03-15',
      expectedHarvest: '2024-07-15',
      healthStatus: 'Good',
      lastInspection: '2024-01-08',
      diseaseCount: 2,
      yieldPrediction: 85,
      location: { latitude: -1.2921, longitude: 36.8219 },
    },
    {
      id: '2',
      name: 'South Field B',
      area: 18.2,
      cropType: 'Maize',
      plantingDate: '2024-03-20',
      expectedHarvest: '2024-07-20',
      healthStatus: 'Excellent',
      lastInspection: '2024-01-07',
      diseaseCount: 0,
      yieldPrediction: 95,
      location: { latitude: -1.2925, longitude: 36.8225 },
    },
    {
      id: '3',
      name: 'East Field C',
      area: 32.1,
      cropType: 'Maize',
      plantingDate: '2024-03-10',
      expectedHarvest: '2024-07-10',
      healthStatus: 'Fair',
      lastInspection: '2024-01-06',
      diseaseCount: 5,
      yieldPrediction: 70,
      location: { latitude: -1.2918, longitude: 36.8230 },
    },
  ]);

  const [farmStats, setFarmStats] = useState({
    totalArea: 75.8,
    totalFields: 3,
    averageHealth: 83,
    totalDiseases: 7,
    expectedYield: 2850, // kg
    weatherAlert: 'Moderate rain expected this week',
  });

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'Excellent': return '#28a745';
      case 'Good': return '#20c997';
      case 'Fair': return '#ffc107';
      case 'Poor': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'Excellent': return 'checkmark-circle';
      case 'Good': return 'checkmark';
      case 'Fair': return 'warning';
      case 'Poor': return 'alert-circle';
      default: return 'help-circle';
    }
  };

  const handleFieldPress = (field: Field) => {
    setSelectedField(field.id);
    Alert.alert(
      `${field.name} Details`,
      `Area: ${field.area} acres\nHealth: ${field.healthStatus}\nDiseases: ${field.diseaseCount}\nYield Prediction: ${field.yieldPrediction}%`,
      [
        { text: 'Inspect Field', onPress: () => navigation.navigate('Camera') },
        { text: 'View History', onPress: () => navigation.navigate('History') },
        { text: 'Close' },
      ]
    );
  };

  const handleAddField = () => {
    Alert.alert(
      'Add New Field',
      'This feature would allow you to add a new field to track.',
      [{ text: 'OK' }]
    );
  };

  const handleWeatherAlert = () => {
    Alert.alert(
      'Weather Alert',
      farmStats.weatherAlert + '\n\nRecommendation: Monitor fields closely and ensure proper drainage.',
      [{ text: 'OK' }]
    );
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={[theme.colors.primary, '#20c997']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <View style={styles.headerInfo}>
            <Text style={styles.headerTitle}>Farm Tracker</Text>
            <Text style={styles.headerSubtitle}>
              {user?.farmName || 'My Farm'} - {farmStats.totalArea} acres
            </Text>
          </View>
          <Menu
            visible={menuVisible}
            onDismiss={() => setMenuVisible(false)}
            anchor={
              <IconButton
                icon="dots-vertical"
                iconColor="white"
                size={24}
                onPress={() => setMenuVisible(true)}
              />
            }
          >
            <Menu.Item onPress={() => {}} title="Export Data" />
            <Menu.Item onPress={() => {}} title="Settings" />
            <Menu.Item onPress={() => {}} title="Help" />
          </Menu>
        </View>
      </LinearGradient>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Farm Overview Stats */}
        <Animatable.View animation="fadeInUp" delay={200} style={styles.statsSection}>
          <Text style={styles.sectionTitle}>Farm Overview</Text>
          <View style={styles.statsGrid}>
            <Surface style={styles.statCard} elevation={3}>
              <Ionicons name="resize" size={24} color={theme.colors.primary} />
              <Text style={styles.statNumber}>{farmStats.totalArea}</Text>
              <Text style={styles.statLabel}>Total Acres</Text>
            </Surface>

            <Surface style={styles.statCard} elevation={3}>
              <Ionicons name="grid" size={24} color="#007bff" />
              <Text style={styles.statNumber}>{farmStats.totalFields}</Text>
              <Text style={styles.statLabel}>Active Fields</Text>
            </Surface>

            <Surface style={styles.statCard} elevation={3}>
              <Ionicons name="heart" size={24} color="#28a745" />
              <Text style={styles.statNumber}>{farmStats.averageHealth}%</Text>
              <Text style={styles.statLabel}>Avg Health</Text>
            </Surface>

            <Surface style={styles.statCard} elevation={3}>
              <Ionicons name="trending-up" size={24} color="#ffc107" />
              <Text style={styles.statNumber}>{farmStats.expectedYield}</Text>
              <Text style={styles.statLabel}>Expected Yield (kg)</Text>
            </Surface>
          </View>
        </Animatable.View>

        {/* Weather Alert */}
        <Animatable.View animation="fadeInUp" delay={400} style={styles.alertSection}>
          <Card style={styles.weatherCard} onPress={handleWeatherAlert}>
            <Card.Content style={styles.weatherContent}>
              <View style={styles.weatherHeader}>
                <Ionicons name="cloud-outline" size={24} color="#007bff" />
                <Text style={styles.weatherTitle}>Weather Alert</Text>
              </View>
              <Text style={styles.weatherText}>{farmStats.weatherAlert}</Text>
              <Text style={styles.weatherAction}>Tap for recommendations</Text>
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Fields List */}
        <Animatable.View animation="fadeInUp" delay={600} style={styles.fieldsSection}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Field Management</Text>
            <Button
              mode="text"
              onPress={handleAddField}
              textColor={theme.colors.primary}
              icon="plus"
            >
              Add Field
            </Button>
          </View>

          {fields.map((field, index) => (
            <Animatable.View
              key={field.id}
              animation="fadeInUp"
              delay={700 + index * 100}
              style={styles.fieldCard}
            >
              <Card style={styles.card} onPress={() => handleFieldPress(field)}>
                <Card.Content style={styles.fieldContent}>
                  <View style={styles.fieldHeader}>
                    <View style={styles.fieldInfo}>
                      <Text style={styles.fieldName}>{field.name}</Text>
                      <Text style={styles.fieldDetails}>
                        {field.area} acres • {field.cropType}
                      </Text>
                      <Text style={styles.fieldDate}>
                        Planted: {new Date(field.plantingDate).toLocaleDateString()}
                      </Text>
                    </View>
                    
                    <View style={styles.fieldStatus}>
                      <Chip
                        style={[
                          styles.healthChip,
                          { backgroundColor: getHealthColor(field.healthStatus) }
                        ]}
                        textStyle={styles.healthText}
                      >
                        <Ionicons
                          name={getHealthIcon(field.healthStatus) as any}
                          size={12}
                          color="white"
                        />
                        {' '}{field.healthStatus}
                      </Chip>
                    </View>
                  </View>

                  <Divider style={styles.divider} />

                  <View style={styles.fieldMetrics}>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricLabel}>Yield Prediction</Text>
                      <ProgressBar
                        progress={field.yieldPrediction / 100}
                        color={field.yieldPrediction > 80 ? '#28a745' : field.yieldPrediction > 60 ? '#ffc107' : '#dc3545'}
                        style={styles.progressBar}
                      />
                      <Text style={styles.metricValue}>{field.yieldPrediction}%</Text>
                    </View>

                    <View style={styles.metricRow}>
                      <View style={styles.metricSmall}>
                        <Ionicons name="bug" size={16} color="#dc3545" />
                        <Text style={styles.metricSmallText}>
                          {field.diseaseCount} diseases
                        </Text>
                      </View>
                      <View style={styles.metricSmall}>
                        <Ionicons name="calendar" size={16} color="#666" />
                        <Text style={styles.metricSmallText}>
                          Last: {new Date(field.lastInspection).toLocaleDateString()}
                        </Text>
                      </View>
                    </View>
                  </View>
                </Card.Content>
              </Card>
            </Animatable.View>
          ))}
        </Animatable.View>

        {/* Quick Actions */}
        <Animatable.View animation="fadeInUp" delay={1000} style={styles.actionsSection}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.actionButtons}>
            <Surface style={styles.actionCard} elevation={3}>
              <Button
                mode="contained"
                onPress={() => navigation.navigate('Camera')}
                style={styles.actionButton}
                contentStyle={styles.actionButtonContent}
                buttonColor={theme.colors.primary}
                icon="camera"
              >
                Inspect Field
              </Button>
            </Surface>

            <Surface style={styles.actionCard} elevation={3}>
              <Button
                mode="contained"
                onPress={() => navigation.navigate('History')}
                style={styles.actionButton}
                contentStyle={styles.actionButtonContent}
                buttonColor="#007bff"
                icon="chart-line"
              >
                View Analytics
              </Button>
            </Surface>
          </View>
        </Animatable.View>

        <View style={styles.bottomSpacing} />
      </ScrollView>

      {/* Floating Action Button */}
      <FAB
        icon="plus"
        style={styles.fab}
        onPress={handleAddField}
        label="Add Field"
      />
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerInfo: {
    flex: 1,
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
  content: {
    flex: 1,
  },
  statsSection: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statCard: {
    width: (width - 60) / 2,
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 15,
    backgroundColor: 'white',
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
  },
  alertSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  weatherCard: {
    borderRadius: 12,
    backgroundColor: '#e3f2fd',
  },
  weatherContent: {
    padding: 16,
  },
  weatherHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  weatherTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007bff',
    marginLeft: 8,
  },
  weatherText: {
    fontSize: 14,
    color: '#1565c0',
    marginBottom: 8,
  },
  weatherAction: {
    fontSize: 12,
    color: '#1976d2',
    fontStyle: 'italic',
  },
  fieldsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  fieldCard: {
    marginBottom: 15,
  },
  card: {
    borderRadius: 12,
    backgroundColor: 'white',
  },
  fieldContent: {
    padding: 16,
  },
  fieldHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  fieldInfo: {
    flex: 1,
  },
  fieldName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  fieldDetails: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  fieldDate: {
    fontSize: 12,
    color: '#999',
  },
  fieldStatus: {
    alignItems: 'flex-end',
  },
  healthChip: {
    borderRadius: 15,
  },
  healthText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  divider: {
    marginVertical: 15,
  },
  fieldMetrics: {
    gap: 12,
  },
  metricItem: {
    gap: 8,
  },
  metricLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  progressBar: {
    height: 6,
    borderRadius: 3,
  },
  metricValue: {
    fontSize: 12,
    color: '#666',
    textAlign: 'right',
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metricSmall: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metricSmallText: {
    fontSize: 12,
    color: '#666',
  },
  actionsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 15,
  },
  actionCard: {
    flex: 1,
    borderRadius: 12,
  },
  actionButton: {
    borderRadius: 12,
  },
  actionButtonContent: {
    paddingVertical: 8,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 80,
    backgroundColor: theme.colors.primary,
  },
  bottomSpacing: {
    height: 100,
  },
});

export default FarmTrackerScreen;