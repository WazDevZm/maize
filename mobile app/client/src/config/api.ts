import axios, { AxiosResponse } from 'axios';
import type { 
  DetectionResult, 
  BatchDetectionResult, 
  HealthCheckResponse, 
  DiseasesResponse 
} from '@/types/api';

// API Configuration
const API_BASE_URL = __DEV__ 
  ? 'http://10.0.2.2:8000' // Android emulator
  : 'http://192.168.1.100:8000'; // Change this to your server IP

// For iOS simulator, use: http://localhost:8000
// For physical device, use your computer's IP address

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
export const API_ENDPOINTS = {
  HEALTH: '/health',
  DISEASES: '/diseases',
  DETECT: '/detect',
  DETECT_BATCH: '/detect-batch',
} as const;

// API functions
export const apiService = {
  // Health check
  checkHealth: async (): Promise<HealthCheckResponse> => {
    try {
      const response: AxiosResponse<HealthCheckResponse> = await api.get(API_ENDPOINTS.HEALTH);
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  // Get all diseases information
  getDiseases: async (): Promise<DiseasesResponse> => {
    try {
      const response: AxiosResponse<DiseasesResponse> = await api.get(API_ENDPOINTS.DISEASES);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch diseases: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  // Detect disease in single image
  detectDisease: async (imageUri: string, confidence: number = 0.25): Promise<DetectionResult> => {
    try {
      const formData = new FormData();
      
      // Create file object from image URI
      const filename = imageUri.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';
      
      formData.append('file', {
        uri: imageUri,
        name: filename,
        type: type,
      } as any);
      
      formData.append('confidence', confidence.toString());

      const response: AxiosResponse<DetectionResult> = await api.post(
        API_ENDPOINTS.DETECT, 
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      return response.data;
    } catch (error) {
      throw new Error(`Disease detection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },

  // Detect diseases in multiple images
  detectBatch: async (imageUris: string[], confidence: number = 0.25): Promise<BatchDetectionResult> => {
    try {
      const formData = new FormData();
      
      imageUris.forEach((imageUri, index) => {
        const filename = imageUri.split('/').pop() || `image_${index}.jpg`;
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';
        
        formData.append('files', {
          uri: imageUri,
          name: filename,
          type: type,
        } as any);
      });
      
      formData.append('confidence', confidence.toString());

      const response: AxiosResponse<BatchDetectionResult> = await api.post(
        API_ENDPOINTS.DETECT_BATCH, 
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      return response.data;
    } catch (error) {
      throw new Error(`Batch detection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  },
};

export default api;