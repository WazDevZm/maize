import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
} from 'react-native';
import {
  Text,
  TextInput,
  Button,
  Card,
  Title,
  Paragraph,
  Chip,
  Surface,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';

import { theme } from '../theme/theme';

const { width, height } = Dimensions.get('window');

interface AuthScreenProps {
  onLogin: (user: any) => void;
}

const AuthScreen: React.FC<AuthScreenProps> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('farmer@maize.com');
  const [password, setPassword] = useState('password123');
  const [name, setName] = useState('');
  const [farmName, setFarmName] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAuth = async () => {
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      if (isLogin) {
        // Dummy login validation
        if (email === 'farmer@maize.com' && password === 'password123') {
          onLogin({
            id: '1',
            name: 'John Farmer',
            email: email,
            farmName: 'Green Valley Farm',
            avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face',
          });
        } else if (email === 'manager@farm.com' && password === 'manager123') {
          onLogin({
            id: '2',
            name: 'Sarah Manager',
            email: email,
            farmName: 'Sunrise Agricultural Co.',
            avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face',
          });
        } else if (email === 'expert@agro.com' && password === 'expert123') {
          onLogin({
            id: '3',
            name: 'Dr. Michael Expert',
            email: email,
            farmName: 'Agricultural Research Center',
            avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
          });
        } else {
          Alert.alert('Login Failed', 'Invalid credentials. Try one of the demo accounts below.');
        }
      } else {
        // Dummy registration
        onLogin({
          id: '4',
          name: name || 'New Farmer',
          email: email,
          farmName: farmName || 'My Farm',
          avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
        });
      }
      setLoading(false);
    }, 1500);
  };

  const demoCredentials = [
    { 
      label: 'Demo Farmer', 
      email: 'farmer@maize.com', 
      password: 'password123',
      description: 'Basic farmer account'
    },
    { 
      label: 'Farm Manager', 
      email: 'manager@farm.com', 
      password: 'manager123',
      description: 'Farm management account'
    },
    { 
      label: 'Agronomist', 
      email: 'expert@agro.com', 
      password: 'expert123',
      description: 'Agricultural expert account'
    },
  ];

  return (
    <View style={styles.container}>
      {/* Background Image with Gradient Overlay */}
      <ImageBackground
        source={{
          uri: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=800&h=1200&fit=crop'
        }}
        style={styles.backgroundImage}
        resizeMode="cover"
      >
        <LinearGradient
          colors={[
            'rgba(255, 255, 255, 0.95)',
            'rgba(255, 255, 255, 0.85)',
            'rgba(46, 139, 87, 0.1)',
            'rgba(46, 139, 87, 0.3)',
          ]}
          style={styles.gradient}
        />
      </ImageBackground>

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <Animatable.View animation="fadeInDown" duration={1000} style={styles.header}>
            <View style={styles.logoContainer}>
              <LinearGradient
                colors={[theme.colors.primary, '#20c997']}
                style={styles.logoGradient}
              >
                <Ionicons name="leaf" size={40} color="white" />
              </LinearGradient>
            </View>
            <Title style={styles.appTitle}>🌽 Maize Disease Detector</Title>
            <Text style={styles.tagline}>Smart Solutions for Modern Farmers</Text>
            <Paragraph style={styles.subtitle}>
              Empowering farmers with AI-powered disease detection for better crop health
            </Paragraph>
          </Animatable.View>

          {/* Auth Card */}
          <Animatable.View animation="fadeInUp" delay={300} duration={800}>
            <Card style={styles.authCard}>
              <Card.Content style={styles.cardContent}>
                {/* Toggle Buttons */}
                <View style={styles.toggleContainer}>
                  <Button
                    mode={isLogin ? 'contained' : 'outlined'}
                    onPress={() => setIsLogin(true)}
                    style={[styles.toggleButton, isLogin && styles.activeToggle]}
                    labelStyle={styles.toggleLabel}
                  >
                    Sign In
                  </Button>
                  <Button
                    mode={!isLogin ? 'contained' : 'outlined'}
                    onPress={() => setIsLogin(false)}
                    style={[styles.toggleButton, !isLogin && styles.activeToggle]}
                    labelStyle={styles.toggleLabel}
                  >
                    Sign Up
                  </Button>
                </View>

                {/* Form Fields */}
                <View style={styles.formContainer}>
                  {!isLogin && (
                    <>
                      <TextInput
                        label="Full Name"
                        value={name}
                        onChangeText={setName}
                        style={styles.input}
                        mode="outlined"
                        left={<TextInput.Icon icon="account" />}
                      />
                      <TextInput
                        label="Farm Name"
                        value={farmName}
                        onChangeText={setFarmName}
                        style={styles.input}
                        mode="outlined"
                        left={<TextInput.Icon icon="home-variant" />}
                      />
                    </>
                  )}
                  
                  <TextInput
                    label="Email"
                    value={email}
                    onChangeText={setEmail}
                    style={styles.input}
                    mode="outlined"
                    keyboardType="email-address"
                    autoCapitalize="none"
                    left={<TextInput.Icon icon="email" />}
                  />
                  
                  <TextInput
                    label="Password"
                    value={password}
                    onChangeText={setPassword}
                    style={styles.input}
                    mode="outlined"
                    secureTextEntry
                    left={<TextInput.Icon icon="lock" />}
                  />
                </View>

                {/* Demo Credentials */}
                {isLogin && (
                  <Animatable.View animation="fadeIn" delay={500}>
                    <Text style={styles.demoTitle}>🎭 Demo Accounts (Tap to use):</Text>
                    <View style={styles.demoContainer}>
                      {demoCredentials.map((cred, index) => (
                        <Surface key={index} style={styles.demoCard} elevation={2}>
                          <Chip
                            style={styles.demoChip}
                            textStyle={styles.demoChipText}
                            onPress={() => {
                              setEmail(cred.email);
                              setPassword(cred.password);
                            }}
                          >
                            {cred.label}
                          </Chip>
                          <Text style={styles.demoDescription}>{cred.description}</Text>
                          <Text style={styles.demoCredText}>
                            {cred.email} / {cred.password}
                          </Text>
                        </Surface>
                      ))}
                    </View>
                  </Animatable.View>
                )}

                {/* Auth Button */}
                <Button
                  mode="contained"
                  onPress={handleAuth}
                  loading={loading}
                  disabled={loading}
                  style={styles.authButton}
                  contentStyle={styles.authButtonContent}
                >
                  {isLogin ? '🚀 Sign In & Start Detecting' : '🌱 Create Account'}
                </Button>

                {/* Features Preview */}
                <View style={styles.featuresContainer}>
                  <Text style={styles.featuresTitle}>✨ What you'll get:</Text>
                  <View style={styles.featuresList}>
                    <View style={styles.featureItem}>
                      <Ionicons name="camera" size={16} color={theme.colors.primary} />
                      <Text style={styles.featureText}>AI Disease Detection (99.5% accuracy)</Text>
                    </View>
                    <View style={styles.featureItem}>
                      <Ionicons name="analytics" size={16} color={theme.colors.primary} />
                      <Text style={styles.featureText}>Detailed Health Reports</Text>
                    </View>
                    <View style={styles.featureItem}>
                      <Ionicons name="medical" size={16} color={theme.colors.primary} />
                      <Text style={styles.featureText}>Treatment Recommendations</Text>
                    </View>
                    <View style={styles.featureItem}>
                      <Ionicons name="time" size={16} color={theme.colors.primary} />
                      <Text style={styles.featureText}>Detection History & Trends</Text>
                    </View>
                  </View>
                </View>
              </Card.Content>
            </Card>
          </Animatable.View>

          {/* Footer */}
          <Animatable.View animation="fadeIn" delay={800} style={styles.footer}>
            <Text style={styles.footerText}>
              🌾 Trusted by farmers worldwide for accurate disease detection
            </Text>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>99.5%</Text>
                <Text style={styles.statLabel}>Accuracy</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>4 Types</Text>
                <Text style={styles.statLabel}>Diseases</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>Real-time</Text>
                <Text style={styles.statLabel}>Detection</Text>
              </View>
            </View>
          </Animatable.View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundImage: {
    position: 'absolute',
    width: width,
    height: height,
  },
  gradient: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  logoContainer: {
    marginBottom: 20,
  },
  logoGradient: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  appTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: theme.colors.primary,
    marginBottom: 5,
    textAlign: 'center',
  },
  tagline: {
    fontSize: 16,
    color: theme.colors.primary,
    fontWeight: '600',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
  },
  authCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 20,
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.15,
    shadowRadius: 20,
    marginBottom: 30,
  },
  cardContent: {
    padding: 25,
  },
  toggleContainer: {
    flexDirection: 'row',
    marginBottom: 25,
    backgroundColor: '#f5f5f5',
    borderRadius: 25,
    padding: 4,
  },
  toggleButton: {
    flex: 1,
    marginHorizontal: 2,
    borderRadius: 20,
  },
  activeToggle: {
    elevation: 2,
  },
  toggleLabel: {
    fontSize: 16,
    fontWeight: '600',
  },
  formContainer: {
    marginBottom: 20,
  },
  input: {
    marginBottom: 15,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
  },
  demoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
    marginBottom: 15,
    textAlign: 'center',
  },
  demoContainer: {
    marginBottom: 20,
  },
  demoCard: {
    padding: 12,
    borderRadius: 12,
    marginBottom: 10,
    backgroundColor: 'rgba(46, 139, 87, 0.05)',
  },
  demoChip: {
    backgroundColor: 'rgba(46, 139, 87, 0.1)',
    borderColor: theme.colors.primary,
    alignSelf: 'flex-start',
    marginBottom: 5,
  },
  demoChipText: {
    fontSize: 12,
    color: theme.colors.primary,
    fontWeight: '600',
  },
  demoDescription: {
    fontSize: 12,
    color: '#666',
    marginBottom: 3,
  },
  demoCredText: {
    fontSize: 11,
    color: '#888',
    fontFamily: 'monospace',
  },
  authButton: {
    borderRadius: 25,
    marginBottom: 25,
    backgroundColor: theme.colors.primary,
  },
  authButtonContent: {
    paddingVertical: 8,
  },
  featuresContainer: {
    marginTop: 10,
  },
  featuresTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
    marginBottom: 15,
    textAlign: 'center',
  },
  featuresList: {
    gap: 8,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  featureText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 8,
  },
  footer: {
    alignItems: 'center',
  },
  footerText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 20,
    textAlign: 'center',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
});

export default AuthScreen;