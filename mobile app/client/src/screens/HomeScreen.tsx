import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  Alert,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Surface,
  Text,
  Chip,
  ActivityIndicator,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

import { theme } from '@/theme/theme';
import { apiService } from '@/config/api';
import type { RootTabParamList, HealthCheckResponse } from '@/types';

const { width } = Dimensions.get('window');

type Props = BottomTabScreenProps<RootTabParamList, 'Home'>;

const HomeScreen: React.FC<Props> = ({ navigation }) => {
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [healthData, setHealthData] = useState<HealthCheckResponse | null>(null);

  useEffect(() => {
    checkServerHealth();
  }, []);

  const checkServerHealth = async (): Promise<void> => {
    try {
      setServerStatus('checking');
      const health = await apiService.checkHealth();
      setHealthData(health);
      setServerStatus('online');
    } catch (error) {
      console.error('Health check failed:', error);
      setServerStatus('offline');
      Alert.alert(
        'Server Offline',
        'Unable to connect to the detection server. Please ensure the server is running.',
        [{ text: 'Retry', onPress: checkServerHealth }]
      );
    }
  };

  const navigateToCamera = (): void => {
    if (serverStatus === 'offline') {
      Alert.alert(
        'Server Offline',
        'Please ensure the server is running before taking photos.',
        [{ text: 'Check Again', onPress: checkServerHealth }]
      );
      return;
    }
    navigation.navigate('Camera');
  };

  const navigateToInfo = (): void => {
    navigation.navigate('Info');
  };

  const navigateToHistory = (): void => {
    navigation.navigate('History');
  };

  const renderServerStatus = (): JSX.Element => {
    const getStatusColor = () => {
      switch (serverStatus) {
        case 'online': return theme.colors.success;
        case 'offline': return theme.colors.error;
        default: return theme.colors.warning;
      }
    };

    const getStatusText = () => {
      switch (serverStatus) {
        case 'online': return 'Server Online';
        case 'offline': return 'Server Offline';
        default: return 'Checking...';
      }
    };

    return (
      <Surface style={styles.statusCard} elevation={2}>
        <View style={styles.statusRow}>
          <View style={styles.statusInfo}>
            <Text style={styles.statusTitle}>Server Status</Text>
            <View style={styles.statusIndicator}>
              {serverStatus === 'checking' ? (
                <ActivityIndicator size="small" color={theme.colors.primary} />
              ) : (
                <View style={[styles.statusDot, { backgroundColor: getStatusColor() }]} />
              )}
              <Text style={[styles.statusText, { color: getStatusColor() }]}>
                {getStatusText()}
              </Text>
            </View>
            {healthData && (
              <Text style={styles.modelStatus}>
                Model: {healthData.model_status === 'loaded' ? '✅ Ready' : '❌ Not Ready'}
              </Text>
            )}
          </View>
          <Button
            mode="outlined"
            onPress={checkServerHealth}
            disabled={serverStatus === 'checking'}
            style={styles.refreshButton}
          >
            Refresh
          </Button>
        </View>
      </Surface>
    );
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Hero Section */}
      <LinearGradient
        colors={[theme.colors.primary, '#1e6b47']}
        style={styles.heroSection}
      >
        <Animatable.View animation="fadeInDown" duration={1000}>
          <Text style={styles.heroTitle}>🌽 Maize Disease Detector</Text>
          <Text style={styles.heroSubtitle}>
            AI-Powered Plant Health Analysis
          </Text>
          <Text style={styles.heroDescription}>
            Detect maize diseases with 99.5% accuracy using advanced computer vision
          </Text>
        </Animatable.View>
      </LinearGradient>

      <View style={styles.content}>
        {/* Server Status */}
        <Animatable.View animation="fadeInUp" delay={200}>
          {renderServerStatus()}
        </Animatable.View>

        {/* Quick Actions */}
        <Animatable.View animation="fadeInUp" delay={400}>
          <Title style={styles.sectionTitle}>Quick Actions</Title>
          <View style={styles.actionGrid}>
            <Card style={styles.actionCard} onPress={navigateToCamera}>
              <Card.Content style={styles.actionContent}>
                <Ionicons name="camera" size={40} color={theme.colors.primary} />
                <Title style={styles.actionTitle}>Detect Disease</Title>
                <Paragraph style={styles.actionDescription}>
                  Take a photo or upload an image to detect diseases
                </Paragraph>
              </Card.Content>
            </Card>

            <Card style={styles.actionCard} onPress={navigateToInfo}>
              <Card.Content style={styles.actionContent}>
                <Ionicons name="information-circle" size={40} color={theme.colors.info} />
                <Title style={styles.actionTitle}>Disease Info</Title>
                <Paragraph style={styles.actionDescription}>
                  Learn about maize diseases and treatments
                </Paragraph>
              </Card.Content>
            </Card>
          </View>
        </Animatable.View>

        {/* Features */}
        <Animatable.View animation="fadeInUp" delay={600}>
          <Title style={styles.sectionTitle}>Key Features</Title>
          <Card style={styles.featureCard}>
            <Card.Content>
              <View style={styles.featureRow}>
                <Ionicons name="flash" size={24} color={theme.colors.success} />
                <View style={styles.featureText}>
                  <Text style={styles.featureTitle}>Real-time Detection</Text>
                  <Text style={styles.featureDescription}>
                    Instant disease identification with high accuracy
                  </Text>
                </View>
              </View>
              
              <View style={styles.featureRow}>
                <Ionicons name="shield-checkmark" size={24} color={theme.colors.success} />
                <View style={styles.featureText}>
                  <Text style={styles.featureTitle}>99.5% Accuracy</Text>
                  <Text style={styles.featureDescription}>
                    Advanced YOLOv8 model trained on maize diseases
                  </Text>
                </View>
              </View>
              
              <View style={styles.featureRow}>
                <Ionicons name="medical" size={24} color={theme.colors.success} />
                <View style={styles.featureText}>
                  <Text style={styles.featureTitle}>Treatment Recommendations</Text>
                  <Text style={styles.featureDescription}>
                    Get specific treatment advice for detected diseases
                  </Text>
                </View>
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>

        {/* Supported Diseases */}
        <Animatable.View animation="fadeInUp" delay={800}>
          <Title style={styles.sectionTitle}>Supported Diseases</Title>
          <View style={styles.chipContainer}>
            <Chip icon="leaf" style={[styles.chip, { backgroundColor: theme.colors.success + '20' }]}>
              Healthy Leaves
            </Chip>
            <Chip icon="alert-circle" style={[styles.chip, { backgroundColor: theme.colors.warning + '20' }]}>
              Grey Leaf Spots
            </Chip>
            <Chip icon="close-circle" style={[styles.chip, { backgroundColor: theme.colors.error + '20' }]}>
              Leaf Blight
            </Chip>
            <Chip icon="bug" style={[styles.chip, { backgroundColor: theme.colors.error + '20' }]}>
              Maize Streak Virus
            </Chip>
          </View>
        </Animatable.View>

        {/* Get Started Button */}
        <Animatable.View animation="fadeInUp" delay={1000}>
          <Button
            mode="contained"
            onPress={navigateToCamera}
            style={styles.getStartedButton}
            contentStyle={styles.getStartedButtonContent}
            disabled={serverStatus === 'offline'}
          >
            🚀 Start Disease Detection
          </Button>
        </Animatable.View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  heroSection: {
    padding: 24,
    paddingTop: 40,
    paddingBottom: 60,
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 18,
    color: 'white',
    textAlign: 'center',
    opacity: 0.9,
    marginBottom: 12,
  },
  heroDescription: {
    fontSize: 14,
    color: 'white',
    textAlign: 'center',
    opacity: 0.8,
    lineHeight: 20,
  },
  content: {
    padding: 16,
    marginTop: -30,
  },
  statusCard: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 24,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusInfo: {
    flex: 1,
  },
  statusTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '500',
  },
  modelStatus: {
    fontSize: 12,
    color: theme.colors.onSurface,
    opacity: 0.7,
  },
  refreshButton: {
    marginLeft: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    color: theme.colors.onBackground,
  },
  actionGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  actionCard: {
    flex: 1,
    marginHorizontal: 4,
    borderRadius: 12,
  },
  actionContent: {
    alignItems: 'center',
    padding: 16,
  },
  actionTitle: {
    fontSize: 16,
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 4,
  },
  actionDescription: {
    fontSize: 12,
    textAlign: 'center',
    opacity: 0.7,
  },
  featureCard: {
    borderRadius: 12,
    marginBottom: 24,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  featureText: {
    marginLeft: 16,
    flex: 1,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  featureDescription: {
    fontSize: 14,
    opacity: 0.7,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 24,
  },
  chip: {
    margin: 4,
  },
  getStartedButton: {
    borderRadius: 25,
    marginBottom: 24,
  },
  getStartedButtonContent: {
    paddingVertical: 8,
  },
});

export default HomeScreen;