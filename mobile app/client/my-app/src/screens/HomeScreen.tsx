import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  Alert,
  RefreshControl,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Surface,
  Avatar,
  Chip,
  ProgressBar,
  Divider,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { StackNavigationProp } from '@react-navigation/stack';

import { theme } from '../theme/theme';
import { User, RootStackParamList } from '../types';

const { width } = Dimensions.get('window');

interface HomeScreenProps {
  user?: User | null;
}

type HomeScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Home'>;

const HomeScreen: React.FC<HomeScreenProps> = ({ user }) => {
  const navigation = useNavigation<HomeScreenNavigationProp>();
  const [refreshing, setRefreshing] = useState(false);
  const [recentDetections, setRecentDetections] = useState([
    {
      id: '1',
      date: '2024-01-08',
      result: 'Healthy',
      confidence: 98.5,
      severity: 'None',
      image: 'leaf1.jpg'
    },
    {
      id: '2',
      date: '2024-01-07',
      result: 'Grey Leaf Spots',
      confidence: 94.2,
      severity: 'Medium',
      image: 'leaf2.jpg'
    },
    {
      id: '3',
      date: '2024-01-06',
      result: 'Healthy',
      confidence: 96.8,
      severity: 'None',
      image: 'leaf3.jpg'
    }
  ]);

  const [farmStats, setFarmStats] = useState({
    totalScans: 127,
    healthyPlants: 89,
    diseasedPlants: 38,
    accuracyRate: 99.5
  });

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  const handleQuickScan = () => {
    navigation.navigate('Camera');
  };

  const handleViewHistory = () => {
    navigation.navigate('History');
  };

  const handleViewDiseaseInfo = () => {
    navigation.navigate('Info');
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return '#dc3545';
      case 'medium': return '#ffc107';
      case 'low': return '#28a745';
      default: return '#28a745';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'alert-circle';
      case 'medium': return 'warning';
      case 'low': return 'checkmark-circle';
      default: return 'checkmark-circle';
    }
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
      showsVerticalScrollIndicator={false}
    >
      {/* Header with Gradient */}
      <LinearGradient
        colors={[theme.colors.primary, '#20c997']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <View style={styles.userInfo}>
            <Avatar.Image
              size={60}
              source={{
                uri: user?.avatar || 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face'
              }}
              style={styles.avatar}
            />
            <View style={styles.userDetails}>
              <Text style={styles.welcomeText}>Welcome back,</Text>
              <Text style={styles.userName}>{user?.name || 'Farmer'}</Text>
              <Text style={styles.farmName}>{user?.farmName || 'Your Farm'}</Text>
            </View>
          </View>
          
          <View style={styles.weatherInfo}>
            <Ionicons name="sunny" size={24} color="white" />
            <Text style={styles.weatherText}>28°C</Text>
          </View>
        </View>
      </LinearGradient>

      {/* Quick Actions */}
      <Animatable.View animation="fadeInUp" delay={200} style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.actionButtons}>
          <Surface style={styles.actionCard} elevation={3}>
            <Button
              mode="contained"
              onPress={handleQuickScan}
              style={styles.actionButton}
              contentStyle={styles.actionButtonContent}
              buttonColor={theme.colors.primary}
            >
              <Ionicons name="camera" size={24} color="white" />
              {'\n'}Scan Leaf
            </Button>
          </Surface>

          <Surface style={styles.actionCard} elevation={3}>
            <Button
              mode="contained"
              onPress={handleViewHistory}
              style={styles.actionButton}
              contentStyle={styles.actionButtonContent}
              buttonColor="#007bff"
            >
              <Ionicons name="time" size={24} color="white" />
              {'\n'}History
            </Button>
          </Surface>

          <Surface style={styles.actionCard} elevation={3}>
            <Button
              mode="contained"
              onPress={handleViewDiseaseInfo}
              style={styles.actionButton}
              contentStyle={styles.actionButtonContent}
              buttonColor="#ffc107"
            >
              <Ionicons name="information-circle" size={24} color="white" />
              {'\n'}Disease Info
            </Button>
          </Surface>
        </View>
      </Animatable.View>

      {/* Farm Statistics */}
      <Animatable.View animation="fadeInUp" delay={400} style={styles.statsSection}>
        <Text style={styles.sectionTitle}>Farm Health Overview</Text>
        <Card style={styles.statsCard}>
          <Card.Content>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{farmStats.totalScans}</Text>
                <Text style={styles.statLabel}>Total Scans</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={[styles.statNumber, { color: theme.colors.success }]}>
                  {farmStats.healthyPlants}
                </Text>
                <Text style={styles.statLabel}>Healthy Plants</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={[styles.statNumber, { color: theme.colors.warning }]}>
                  {farmStats.diseasedPlants}
                </Text>
                <Text style={styles.statLabel}>Need Attention</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={[styles.statNumber, { color: theme.colors.primary }]}>
                  {farmStats.accuracyRate}%
                </Text>
                <Text style={styles.statLabel}>Accuracy</Text>
              </View>
            </View>
            
            <Divider style={styles.divider} />
            
            <View style={styles.healthProgress}>
              <Text style={styles.progressLabel}>Overall Farm Health</Text>
              <ProgressBar
                progress={farmStats.healthyPlants / farmStats.totalScans}
                color={theme.colors.success}
                style={styles.progressBar}
              />
              <Text style={styles.progressText}>
                {Math.round((farmStats.healthyPlants / farmStats.totalScans) * 100)}% Healthy
              </Text>
            </View>
          </Card.Content>
        </Card>
      </Animatable.View>

      {/* Recent Detections */}
      <Animatable.View animation="fadeInUp" delay={600} style={styles.recentSection}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Detections</Text>
          <Button
            mode="text"
            onPress={handleViewHistory}
            textColor={theme.colors.primary}
          >
            View All
          </Button>
        </View>

        {recentDetections.map((detection, index) => (
          <Card key={detection.id} style={styles.detectionCard}>
            <Card.Content style={styles.detectionContent}>
              <View style={styles.detectionHeader}>
                <View style={styles.detectionInfo}>
                  <Text style={styles.detectionDate}>{detection.date}</Text>
                  <Text style={styles.detectionResult}>{detection.result}</Text>
                  <Text style={styles.detectionConfidence}>
                    Confidence: {detection.confidence}%
                  </Text>
                </View>
                
                <View style={styles.detectionStatus}>
                  <Chip
                    style={[
                      styles.severityChip,
                      { backgroundColor: getSeverityColor(detection.severity) }
                    ]}
                    textStyle={styles.severityText}
                  >
                    <Ionicons
                      name={getSeverityIcon(detection.severity) as any}
                      size={14}
                      color="white"
                    />
                    {' '}{detection.severity}
                  </Chip>
                </View>
              </View>
            </Card.Content>
          </Card>
        ))}
      </Animatable.View>

      {/* Disease Alert */}
      <Animatable.View animation="fadeInUp" delay={800} style={styles.alertSection}>
        <Card style={[styles.alertCard, { backgroundColor: '#fff3cd' }]}>
          <Card.Content>
            <View style={styles.alertContent}>
              <Ionicons name="warning" size={24} color="#856404" />
              <View style={styles.alertText}>
                <Text style={styles.alertTitle}>Disease Alert</Text>
                <Text style={styles.alertMessage}>
                  Grey Leaf Spots detected in 3 recent scans. Consider preventive measures.
                </Text>
              </View>
            </View>
            <Button
              mode="outlined"
              onPress={() => Alert.alert('Treatment Info', 'Apply fungicides and improve air circulation')}
              style={styles.alertButton}
              textColor="#856404"
            >
              View Treatment
            </Button>
          </Card.Content>
        </Card>
      </Animatable.View>

      {/* Tips Section */}
      <Animatable.View animation="fadeInUp" delay={1000} style={styles.tipsSection}>
        <Text style={styles.sectionTitle}>💡 Today's Tip</Text>
        <Card style={styles.tipCard}>
          <Card.Content>
            <Text style={styles.tipText}>
              🌱 Regular monitoring is key! Scan your maize leaves weekly to catch diseases early 
              and maintain optimal crop health.
            </Text>
          </Card.Content>
        </Card>
      </Animatable.View>

      <View style={styles.bottomSpacing} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  userInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  avatar: {
    marginRight: 15,
  },
  userDetails: {
    flex: 1,
  },
  welcomeText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
  },
  userName: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  farmName: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 14,
  },
  weatherInfo: {
    alignItems: 'center',
  },
  weatherText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 4,
  },
  quickActions: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionCard: {
    flex: 1,
    marginHorizontal: 5,
    borderRadius: 15,
  },
  actionButton: {
    borderRadius: 15,
  },
  actionButtonContent: {
    paddingVertical: 20,
    paddingHorizontal: 10,
  },
  statsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  statsCard: {
    borderRadius: 15,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
  },
  divider: {
    marginVertical: 15,
  },
  healthProgress: {
    alignItems: 'center',
  },
  progressLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  progressBar: {
    width: '100%',
    height: 8,
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: '#666',
    marginTop: 8,
  },
  recentSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  detectionCard: {
    marginBottom: 10,
    borderRadius: 12,
  },
  detectionContent: {
    paddingVertical: 15,
  },
  detectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  detectionInfo: {
    flex: 1,
  },
  detectionDate: {
    fontSize: 12,
    color: '#666',
  },
  detectionResult: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginVertical: 2,
  },
  detectionConfidence: {
    fontSize: 12,
    color: '#666',
  },
  detectionStatus: {
    alignItems: 'flex-end',
  },
  severityChip: {
    borderRadius: 15,
  },
  severityText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  alertSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  alertCard: {
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#ffc107',
  },
  alertContent: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  alertText: {
    flex: 1,
    marginLeft: 15,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#856404',
  },
  alertMessage: {
    fontSize: 14,
    color: '#856404',
    marginTop: 4,
  },
  alertButton: {
    borderColor: '#856404',
  },
  tipsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  tipCard: {
    borderRadius: 12,
    backgroundColor: '#e8f5e8',
  },
  tipText: {
    fontSize: 14,
    color: '#2d5a2d',
    lineHeight: 20,
  },
  bottomSpacing: {
    height: 20,
  },
});

export default HomeScreen;