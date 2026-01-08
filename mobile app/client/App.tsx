import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

// Import screens
import HomeScreen from '@/screens/HomeScreen';
import CameraScreen from '@/screens/CameraScreen';
import HistoryScreen from '@/screens/HistoryScreen';
import InfoScreen from '@/screens/InfoScreen';

// Import theme and types
import { theme } from '@/theme/theme';
import type { RootTabParamList } from '@/types/navigation';

const Tab = createBottomTabNavigator<RootTabParamList>();

export default function App(): JSX.Element {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <PaperProvider theme={theme}>
        <NavigationContainer>
          <StatusBar style="auto" />
          <Tab.Navigator
            screenOptions={({ route }) => ({
              tabBarIcon: ({ focused, color, size }) => {
                let iconName: keyof typeof Ionicons.glyphMap;

                switch (route.name) {
                  case 'Home':
                    iconName = focused ? 'home' : 'home-outline';
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
              tabBarInactiveTintColor: 'gray',
              headerStyle: {
                backgroundColor: theme.colors.primary,
              },
              headerTintColor: '#fff',
              headerTitleStyle: {
                fontWeight: 'bold',
              },
            })}
          >
            <Tab.Screen 
              name="Home" 
              component={HomeScreen}
              options={{ title: '🌽 Maize Disease Detector' }}
            />
            <Tab.Screen 
              name="Camera" 
              component={CameraScreen}
              options={{ title: '📷 Disease Detection' }}
            />
            <Tab.Screen 
              name="History" 
              component={HistoryScreen}
              options={{ title: '📊 Detection History' }}
            />
            <Tab.Screen 
              name="Info" 
              component={InfoScreen}
              options={{ title: 'ℹ️ Disease Information' }}
            />
          </Tab.Navigator>
        </NavigationContainer>
      </PaperProvider>
    </GestureHandlerRootView>
  );
}