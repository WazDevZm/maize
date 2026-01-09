import React, { useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Provider as PaperProvider } from 'react-native-paper';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

// Import screens
import {
  AuthScreen,
  HomeScreen,
  CameraScreen,
  HistoryScreen,
  InfoScreen,
  DetectionResultScreen,
  FarmTrackerScreen,
} from './src/screens';

// Import theme and types
import { theme } from './src/theme/theme';
import { User } from './src/types';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

function MainTabs({ user }: { user: User | null }): React.JSX.Element {
  return (
    <Tab.Navigator
      screenOptions={({ route }: { route: any }) => ({
        tabBarIcon: ({ focused, color, size }: { focused: boolean; color: string; size: number }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'FarmTracker':
              iconName = focused ? 'leaf' : 'leaf-outline';
              break;
            case 'Camera':
              iconName = focused ? 'camera' : 'camera-outline';
              break;
            case 'History':
              iconName = focused ? 'time' : 'time-outline';
              break;
            case 'Info':
              iconName = focused ? 'information-circle' : 'information-circle-outline';
              break;
            default:
              iconName = 'help-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: '#8E8E93',
        tabBarStyle: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          borderTopWidth: 0,
          elevation: 20,
          shadowColor: '#000',
          shadowOffset: { width: 0, height: -2 },
          shadowOpacity: 0.1,
          shadowRadius: 10,
          height: 85,
          paddingBottom: 20,
          paddingTop: 10,
        },
        headerShown: false,
      })}
    >
      <Tab.Screen 
        name="Home" 
        options={{ tabBarLabel: 'Dashboard' }}
      >
        {() => <HomeScreen user={user} />}
      </Tab.Screen>
      <Tab.Screen 
        name="FarmTracker" 
        options={{ tabBarLabel: 'Farm Tracker' }}
      >
        {() => <FarmTrackerScreen user={user} />}
      </Tab.Screen>
      <Tab.Screen 
        name="Camera" 
        component={CameraScreen}
        options={{ tabBarLabel: 'Detect Disease' }}
      />
      <Tab.Screen 
        name="History" 
        component={HistoryScreen}
        options={{ tabBarLabel: 'History' }}
      />
      <Tab.Screen 
        name="Info" 
        component={InfoScreen}
        options={{ tabBarLabel: 'Disease Info' }}
      />
    </Tab.Navigator>
  );
}

export default function App(): React.JSX.Element {
  const [user, setUser] = useState<User | null>(null);

  const handleLogin = (userData: User) => {
    console.log('App: User logged in:', userData);
    setUser(userData);
  };

  console.log('App: Current user state:', user);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <PaperProvider theme={theme}>
        <NavigationContainer>
          <StatusBar style="light" backgroundColor={theme.colors.primary} />
          {!user ? (
            <AuthScreen onLogin={handleLogin} />
          ) : (
            <Stack.Navigator screenOptions={{ headerShown: false }}>
              <Stack.Screen name="Main">
                {() => <MainTabs user={user} />}
              </Stack.Screen>
              <Stack.Screen 
                name="DetectionResult" 
                component={DetectionResultScreen}
                options={{
                  headerShown: true,
                  title: 'Detection Results',
                  headerStyle: { backgroundColor: theme.colors.primary },
                  headerTintColor: '#fff',
                }}
              />
            </Stack.Navigator>
          )}
        </NavigationContainer>
      </PaperProvider>
    </GestureHandlerRootView>
  );
}