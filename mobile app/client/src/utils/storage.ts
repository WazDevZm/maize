import AsyncStorage from '@react-native-async-storage/async-storage';
import type { DetectionHistory } from '@/types';

const STORAGE_KEYS = {
  DETECTION_HISTORY: '@maize_detector_history',
  USER_PREFERENCES: '@maize_detector_preferences',
} as const;

export const saveDetectionHistory = async (detection: DetectionHistory): Promise<void> => {
  try {
    const existingHistory = await getDetectionHistory();
    const updatedHistory = [detection, ...existingHistory].slice(0, 100); // Keep last 100 detections
    
    await AsyncStorage.setItem(
      STORAGE_KEYS.DETECTION_HISTORY,
      JSON.stringify(updatedHistory)
    );
  } catch (error) {
    console.error('Error saving detection history:', error);
    throw new Error('Failed to save detection history');
  }
};

export const getDetectionHistory = async (): Promise<DetectionHistory[]> => {
  try {
    const historyJson = await AsyncStorage.getItem(STORAGE_KEYS.DETECTION_HISTORY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error('Error loading detection history:', error);
    return [];
  }
};

export const clearDetectionHistory = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(STORAGE_KEYS.DETECTION_HISTORY);
  } catch (error) {
    console.error('Error clearing detection history:', error);
    throw new Error('Failed to clear detection history');
  }
};

export const deleteDetectionFromHistory = async (id: string): Promise<void> => {
  try {
    const history = await getDetectionHistory();
    const updatedHistory = history.filter(item => item.id !== id);
    
    await AsyncStorage.setItem(
      STORAGE_KEYS.DETECTION_HISTORY,
      JSON.stringify(updatedHistory)
    );
  } catch (error) {
    console.error('Error deleting detection from history:', error);
    throw new Error('Failed to delete detection from history');
  }
};

interface UserPreferences {
  defaultConfidence: number;
  enableHaptics: boolean;
  enableNotifications: boolean;
}

export const saveUserPreferences = async (preferences: UserPreferences): Promise<void> => {
  try {
    await AsyncStorage.setItem(
      STORAGE_KEYS.USER_PREFERENCES,
      JSON.stringify(preferences)
    );
  } catch (error) {
    console.error('Error saving user preferences:', error);
    throw new Error('Failed to save user preferences');
  }
};

export const getUserPreferences = async (): Promise<UserPreferences> => {
  try {
    const preferencesJson = await AsyncStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
    return preferencesJson ? JSON.parse(preferencesJson) : {
      defaultConfidence: 0.25,
      enableHaptics: true,
      enableNotifications: true,
    };
  } catch (error) {
    console.error('Error loading user preferences:', error);
    return {
      defaultConfidence: 0.25,
      enableHaptics: true,
      enableNotifications: true,
    };
  }
};