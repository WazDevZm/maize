import React from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
} from 'react-native';
import {
  Text,
  Card,
  Button,
} from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import type { StackNavigationProp } from '@react-navigation/stack';

import { theme } from '../theme/theme';
import { User, RootStackParamList } from '../types';

interface HomeScreenProps {
  user?: User | null;
}

type HomeScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Home'>;

const HomeScreen: React.FC<HomeScreenProps> = ({ user }) => {
  const navigation = useNavigation<HomeScreenNavigationProp>();

  console.log('HomeScreen: Rendering with user:', user);

  return (
    <ScrollView style={styles.container}>
      {/* Simple Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Maize Disease Detection</Text>
        <Text style={styles.welcomeText}>Welcome, {user?.name || 'Farmer'}!</Text>
        <Text style={styles.farmText}>Farm: {user?.farmName || 'My Farm'}</Text>
      </View>

      {/* Success Card */}
      <View style={styles.content}>
        <Card style={styles.card}>
          <Card.Content>
            <Text style={styles.successTitle}>🎉 Login Successful!</Text>
            <Text style={styles.successText}>
              You are now logged in to your dashboard
            </Text>
          </Card.Content>
        </Card>

        {/* Navigation Buttons */}
        <View style={styles.buttonContainer}>
          <Button
            mode="contained"
            onPress={() => navigation.navigate('Camera')}
            style={styles.button}
          >
            📷 Scan Leaf
          </Button>

          <Button
            mode="contained"
            onPress={() => navigation.navigate('History')}
            style={styles.button}
          >
            📊 View History
          </Button>

          <Button
            mode="contained"
            onPress={() => navigation.navigate('Info')}
            style={styles.button}
          >
            ℹ️ Disease Info
          </Button>
        </View>

        {/* Simple Stats */}
        <Card style={styles.card}>
          <Card.Content>
            <Text style={styles.statsTitle}>Quick Stats</Text>
            <Text style={styles.statsText}>Total Scans: 127</Text>
            <Text style={styles.statsText}>Healthy Plants: 89</Text>
            <Text style={styles.statsText}>Accuracy: 99.5%</Text>
          </Card.Content>
        </Card>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: theme.colors.primary,
    padding: 30,
    paddingTop: 60,
  },
  headerTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10,
  },
  welcomeText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 5,
  },
  farmText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    textAlign: 'center',
  },
  content: {
    padding: 20,
  },
  card: {
    marginBottom: 20,
    borderRadius: 10,
  },
  successTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: theme.colors.primary,
    textAlign: 'center',
    marginBottom: 10,
  },
  successText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  buttonContainer: {
    marginBottom: 20,
  },
  button: {
    marginBottom: 15,
    borderRadius: 10,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  statsText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
    textAlign: 'center',
  },
});

export default HomeScreen;