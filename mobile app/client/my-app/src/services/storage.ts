import AsyncStorage from '@react-native-async-storage/async-storage';
import { User, HistoryItem, AppSettings } from '../types';

// Storage keys
const STORAGE_KEYS = {
  USER: '@maize_app:user',
  AUTH_TOKEN: '@maize_app:auth_token',
  DETECTION_HISTORY: '@maize_app:detection_history',
  APP_SETTINGS: '@maize_app:app_settings',
  FARM_STATS: '@maize_app:farm_stats',
} as const;

class StorageService {
  // Generic storage methods
  private async setItem<T>(key: string, value: T): Promise<void> {
    try {
      const jsonValue = JSON.stringify(value);
      await AsyncStorage.setItem(key, jsonValue);
    } catch (error) {
      console.error('Error saving to storage:', error);
      throw new Error('Failed to save data');
    }
  }

  private async getItem<T>(key: string): Promise<T | null> {
    try {
      const jsonValue = await AsyncStorage.getItem(key);
      return jsonValue != null ? JSON.parse(jsonValue) : null;
    } catch (error) {
      console.error('Error reading from storage:', error);
      return null;
    }
  }

  private async removeItem(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from storage:', error);
      throw new Error('Failed to remove data');
    }
  }

  // User management
  async saveUser(user: User): Promise<void> {
    await this.setItem(STORAGE_KEYS.USER, user);
  }

  async getUser(): Promise<User | null> {
    return this.getItem<User>(STORAGE_KEYS.USER);
  }

  async removeUser(): Promise<void> {
    await this.removeItem(STORAGE_KEYS.USER);
  }

  // Authentication token
  async saveAuthToken(token: string): Promise<void> {
    await this.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
  }

  async getAuthToken(): Promise<string | null> {
    return this.getItem<string>(STORAGE_KEYS.AUTH_TOKEN);
  }

  async removeAuthToken(): Promise<void> {
    await this.removeItem(STORAGE_KEYS.AUTH_TOKEN);
  }

  // Detection history
  async saveDetectionHistory(history: HistoryItem[]): Promise<void> {
    await this.setItem(STORAGE_KEYS.DETECTION_HISTORY, history);
  }

  async getDetectionHistory(): Promise<HistoryItem[]> {
    const history = await this.getItem<HistoryItem[]>(STORAGE_KEYS.DETECTION_HISTORY);
    return history || [];
  }

  async addDetectionToHistory(detection: HistoryItem): Promise<void> {
    const currentHistory = await this.getDetectionHistory();
    const updatedHistory = [detection, ...currentHistory];
    
    // Keep only the last 100 detections to prevent storage bloat
    const trimmedHistory = updatedHistory.slice(0, 100);
    
    await this.saveDetectionHistory(trimmedHistory);
  }

  async removeDetectionFromHistory(detectionId: string): Promise<void> {
    const currentHistory = await this.getDetectionHistory();
    const updatedHistory = currentHistory.filter(item => item.id !== detectionId);
    await this.saveDetectionHistory(updatedHistory);
  }

  async clearDetectionHistory(): Promise<void> {
    await this.saveDetectionHistory([]);
  }

  // App settings
  async saveAppSettings(settings: AppSettings): Promise<void> {
    await this.setItem(STORAGE_KEYS.APP_SETTINGS, settings);
  }

  async getAppSettings(): Promise<AppSettings> {
    const settings = await this.getItem<AppSettings>(STORAGE_KEYS.APP_SETTINGS);
    
    // Return default settings if none exist
    return settings || {
      notifications: true,
      autoSave: true,
      confidenceThreshold: 0.25,
      language: 'en',
      theme: 'light',
    };
  }

  // Farm statistics
  async saveFarmStats(stats: any): Promise<void> {
    await this.setItem(STORAGE_KEYS.FARM_STATS, stats);
  }

  async getFarmStats(): Promise<any> {
    const stats = await this.getItem(STORAGE_KEYS.FARM_STATS);
    
    // Return default stats if none exist
    return stats || {
      totalScans: 0,
      healthyPlants: 0,
      diseasedPlants: 0,
      accuracyRate: 99.5,
      lastScanDate: null,
    };
  }

  // Utility methods
  async clearAllData(): Promise<void> {
    try {
      await AsyncStorage.multiRemove(Object.values(STORAGE_KEYS));
    } catch (error) {
      console.error('Error clearing all data:', error);
      throw new Error('Failed to clear data');
    }
  }

  async getStorageSize(): Promise<{ keys: number; estimatedSize: string }> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      let totalSize = 0;
      
      for (const key of keys) {
        const value = await AsyncStorage.getItem(key);
        if (value) {
          totalSize += value.length;
        }
      }
      
      // Convert bytes to readable format
      const formatSize = (bytes: number): string => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
      };
      
      return {
        keys: keys.length,
        estimatedSize: formatSize(totalSize),
      };
    } catch (error) {
      console.error('Error getting storage size:', error);
      return { keys: 0, estimatedSize: '0 Bytes' };
    }
  }

  // Export/Import functionality
  async exportData(): Promise<string> {
    try {
      const user = await this.getUser();
      const history = await this.getDetectionHistory();
      const settings = await this.getAppSettings();
      const stats = await this.getFarmStats();
      
      const exportData = {
        user,
        history,
        settings,
        stats,
        exportDate: new Date().toISOString(),
        version: '1.0.0',
      };
      
      return JSON.stringify(exportData, null, 2);
    } catch (error) {
      console.error('Error exporting data:', error);
      throw new Error('Failed to export data');
    }
  }

  async importData(jsonData: string): Promise<void> {
    try {
      const data = JSON.parse(jsonData);
      
      if (data.user) await this.saveUser(data.user);
      if (data.history) await this.saveDetectionHistory(data.history);
      if (data.settings) await this.saveAppSettings(data.settings);
      if (data.stats) await this.saveFarmStats(data.stats);
      
    } catch (error) {
      console.error('Error importing data:', error);
      throw new Error('Failed to import data');
    }
  }
}

// Create and export a singleton instance
export const storageService = new StorageService();

// Export the class for testing or custom instances
export default StorageService;