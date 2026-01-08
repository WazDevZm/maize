import { Alert } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { DetectionResult, HistoryItem } from '../types';

// Date and time utilities
export const formatDate = (date: string | Date): string => {
  const d = new Date(date);
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

export const formatTime = (date: string | Date): string => {
  const d = new Date(date);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const formatDateTime = (date: string | Date): string => {
  const d = new Date(date);
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const getRelativeTime = (date: string | Date): string => {
  const now = new Date();
  const past = new Date(date);
  const diffInSeconds = Math.floor((now.getTime() - past.getTime()) / 1000);

  if (diffInSeconds < 60) return 'Just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
  
  return formatDate(date);
};

// File utilities
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const getFileExtension = (filename: string): string => {
  return filename.split('.').pop()?.toLowerCase() || '';
};

export const isImageFile = (filename: string): boolean => {
  const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'];
  return imageExtensions.includes(getFileExtension(filename));
};

// Validation utilities
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const validatePassword = (password: string): { isValid: boolean; message?: string } => {
  if (password.length < 6) {
    return { isValid: false, message: 'Password must be at least 6 characters long' };
  }
  return { isValid: true };
};

// Disease detection utilities
export const getSeverityColor = (severity: string): string => {
  switch (severity.toLowerCase()) {
    case 'high': return '#dc3545';
    case 'medium': return '#ffc107';
    case 'low': return '#28a745';
    case 'none': return '#28a745';
    default: return '#6c757d';
  }
};

export const getSeverityIcon = (severity: string): string => {
  switch (severity.toLowerCase()) {
    case 'high': return 'alert-circle';
    case 'medium': return 'warning';
    case 'low': return 'checkmark-circle';
    case 'none': return 'checkmark-circle';
    default: return 'information-circle';
  }
};

export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 90) return '#28a745';
  if (confidence >= 70) return '#ffc107';
  return '#dc3545';
};

export const getConfidenceLabel = (confidence: number): string => {
  if (confidence >= 90) return 'High Confidence';
  if (confidence >= 70) return 'Medium Confidence';
  return 'Low Confidence';
};

// Data processing utilities
export const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

// Statistics utilities
export const calculateAccuracy = (predictions: DetectionResult[]): number => {
  if (predictions.length === 0) return 0;
  const totalConfidence = predictions.reduce((sum, pred) => sum + pred.confidence, 0);
  return totalConfidence / predictions.length;
};

export const getHealthPercentage = (history: HistoryItem[]): number => {
  if (history.length === 0) return 0;
  const healthyCount = history.filter(item => item.result === 'Healthy').length;
  return (healthyCount / history.length) * 100;
};

export const getDiseaseDistribution = (history: HistoryItem[]): Record<string, number> => {
  const distribution: Record<string, number> = {};
  
  history.forEach(item => {
    distribution[item.result] = (distribution[item.result] || 0) + 1;
  });
  
  return distribution;
};

// Export utilities
export const exportHistoryToCSV = (history: HistoryItem[]): string => {
  const headers = ['Date', 'Time', 'Disease', 'Confidence', 'Severity', 'Location', 'Treatment'];
  const csvContent = [
    headers.join(','),
    ...history.map(item => [
      item.date,
      item.time,
      item.result,
      item.confidence.toFixed(1),
      item.severity,
      item.location || '',
      `"${item.treatment.replace(/"/g, '""')}"` // Escape quotes in treatment text
    ].join(','))
  ].join('\n');
  
  return csvContent;
};

export const shareFile = async (content: string, filename: string, mimeType: string = 'text/plain'): Promise<void> => {
  try {
    const fileUri = FileSystem.documentDirectory + filename;
    await FileSystem.writeAsStringAsync(fileUri, content);
    
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(fileUri, {
        mimeType,
        dialogTitle: 'Share Detection Data',
      });
    } else {
      Alert.alert('Sharing not available', 'File saved to device storage');
    }
  } catch (error) {
    console.error('Error sharing file:', error);
    Alert.alert('Error', 'Failed to share file');
  }
};

// Error handling utilities
export const handleApiError = (error: any): string => {
  if (error?.response?.data?.message) {
    return error.response.data.message;
  }
  
  if (error?.message) {
    return error.message;
  }
  
  if (typeof error === 'string') {
    return error;
  }
  
  return 'An unexpected error occurred';
};

export const showErrorAlert = (title: string, message: string): void => {
  Alert.alert(title, message, [{ text: 'OK' }]);
};

export const showSuccessAlert = (title: string, message: string): void => {
  Alert.alert(title, message, [{ text: 'OK' }]);
};

// Image utilities
export const resizeImageUri = async (uri: string, maxWidth: number, maxHeight: number): Promise<string> => {
  // This would typically use a library like expo-image-manipulator
  // For now, return the original URI
  return uri;
};

export const compressImage = async (uri: string, quality: number = 0.8): Promise<string> => {
  // This would typically use a library like expo-image-manipulator
  // For now, return the original URI
  return uri;
};

// Network utilities
export const isNetworkError = (error: any): boolean => {
  return error?.code === 'NETWORK_ERROR' || 
         error?.message?.includes('Network') ||
         error?.message?.includes('fetch');
};

export const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> => {
  let lastError: any;
  
  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (i === maxRetries) {
        throw error;
      }
      
      // Exponential backoff
      const delay = baseDelay * Math.pow(2, i);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError;
};