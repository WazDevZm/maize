import { DetectionApiResponse, ApiResponse, HistoryItem } from '../types';

// API Configuration
const API_BASE_URL = __DEV__ ? 'http://localhost:8000' : 'https://your-production-api.com'; // Use localhost for web, 10.0.2.2 for Android emulator
const API_TIMEOUT = 30000; // 30 seconds

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // Generic API request method
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const config: RequestInit = {
        timeout: API_TIMEOUT,
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      };

      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; model_status: string }>> {
    return this.request('/health');
  }

  // Disease detection
  async detectDisease(
    imageUri: string,
    confidence: number = 0.25
  ): Promise<ApiResponse<DetectionApiResponse>> {
    try {
      const formData = new FormData();
      
      // Add the image file
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'maize_leaf.jpg',
      } as any);

      // Add confidence parameter
      formData.append('confidence', confidence.toString());

      const response = await fetch(`${this.baseUrl}/detect`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.status}`);
      }

      const data = await response.json();
      
      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error('Disease detection failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Detection failed',
      };
    }
  }

  // Batch disease detection
  async detectDiseasesBatch(
    imageUris: string[],
    confidence: number = 0.25
  ): Promise<ApiResponse<{ batch_results: any[] }>> {
    try {
      const formData = new FormData();
      
      // Add multiple image files
      imageUris.forEach((uri, index) => {
        formData.append('files', {
          uri,
          type: 'image/jpeg',
          name: `maize_leaf_${index}.jpg`,
        } as any);
      });

      // Add confidence parameter
      formData.append('confidence', confidence.toString());

      const response = await fetch(`${this.baseUrl}/detect-batch`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`Batch detection failed: ${response.status}`);
      }

      const data = await response.json();
      
      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error('Batch detection failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Batch detection failed',
      };
    }
  }

  // Get disease information
  async getDiseases(): Promise<ApiResponse<{ diseases: any; total_diseases: number }>> {
    return this.request('/diseases');
  }

  // Get specific disease information
  async getDiseaseInfo(diseaseName: string): Promise<ApiResponse<{ disease: string; info: any }>> {
    return this.request(`/diseases/${encodeURIComponent(diseaseName)}`);
  }

  // Mock authentication (accepts any valid email/password for demo)
  async login(email: string, password: string): Promise<ApiResponse<{ user: any; token: string }>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Basic validation
    if (!email || !password) {
      return {
        success: false,
        error: 'Email and password are required',
      };
    }

    // Simple email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return {
        success: false,
        error: 'Please enter a valid email address',
      };
    }

    // Password validation
    if (password.length < 6) {
      return {
        success: false,
        error: 'Password must be at least 6 characters long',
      };
    }

    // Predefined users for specific emails
    const predefinedUsers = {
      'farmer@maize.com': {
        id: '1',
        name: 'John Farmer',
        email: 'farmer@maize.com',
        farmName: 'Green Valley Farm',
        role: 'farmer',
        avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face',
      },
      'manager@farm.com': {
        id: '2',
        name: 'Sarah Manager',
        email: 'manager@farm.com',
        farmName: 'Sunrise Agricultural Co.',
        role: 'manager',
        avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face',
      },
      'expert@agro.com': {
        id: '3',
        name: 'Dr. Michael Expert',
        email: 'expert@agro.com',
        farmName: 'Agricultural Research Center',
        role: 'expert',
        avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
      },
    };

    // Check if it's a predefined user
    const predefinedUser = predefinedUsers[email as keyof typeof predefinedUsers];
    if (predefinedUser) {
      return {
        success: true,
        data: {
          user: predefinedUser,
          token: 'dummy_jwt_token_' + Date.now(),
        },
      };
    }

    // For any other valid email/password combination, create a generic user
    const genericUser = {
      id: Date.now().toString(),
      name: email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1),
      email: email,
      farmName: 'My Farm',
      role: 'farmer' as const,
      avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
    };

    return {
      success: true,
      data: {
        user: genericUser,
        token: 'dummy_jwt_token_' + Date.now(),
      },
    };
  }

  // Mock registration
  async register(userData: {
    name: string;
    email: string;
    password: string;
    farmName?: string;
  }): Promise<ApiResponse<{ user: any; token: string }>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    const newUser = {
      id: Date.now().toString(),
      name: userData.name,
      email: userData.email,
      farmName: userData.farmName || 'My Farm',
      role: 'farmer',
      avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
    };

    return {
      success: true,
      data: {
        user: newUser,
        token: 'dummy_jwt_token_' + Date.now(),
      },
    };
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();

// Export the class for testing or custom instances
export default ApiService;