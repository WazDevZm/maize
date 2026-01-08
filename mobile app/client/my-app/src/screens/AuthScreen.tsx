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
import { apiService } from '../services/api';

const { width, height } = Dimensions.get('window');

interface AuthScreenProps {
  onLogin: (user: any) => void;
}

const AuthScreen: React.FC<AuthScreenProps> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [farmName, setFarmName] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAuth = async () => {
    setLoading(true);
    
    try {
      if (isLogin) {
        // Use API service for login
        const response = await apiService.login(email, password);
        
        if (response.success && response.data) {
          onLogin(response.data.user);
        } else {
          Alert.alert('Login Failed', response.error || 'Invalid credentials. Check README.md for demo accounts.');
        }
      } else {
        // Use API service for registration
        const response = await apiService.register({
          name: name || 'New Farmer',
          email,
          password,
          farmName: farmName || 'My Farm',
        });
        
        if (response.success && response.data) {
          onLogin(response.data.user);
        } else {
          Alert.alert('Registration Failed', response.error || 'Failed to create account.');
        }
      }
    } catch (error) {
      Alert.alert('Error', 'Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Background Image with Gradient Overlay */}
      <ImageBackground
        source={{
          uri: 'https://images.unsplash.com/photo-1594771804886-a933bb2d609b?q=80&w=882&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
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
            <Title style={styles.appTitle}>Maize Disease Detector</Title>
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

                {/* Demo Credentials Info */}
                {isLogin && (
                  <Animatable.View animation="fadeIn" delay={500}>
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
                  {isLogin ? 'Sign In & Start Detecting' : '🌱 Create Account'}
                </Button>
              </Card.Content>
            </Card>
          </Animatable.View>

          {/* Footer */}
          <Animatable.View animation="fadeIn" delay={800} style={styles.footer}>
           
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
  demoNote: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
    backgroundColor: 'rgba(46, 139, 87, 0.05)',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
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