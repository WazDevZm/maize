// User types
export interface User {
  id: string;
  name: string;
  email: string;
  farmName?: string;
  avatar?: string;
  role?: 'farmer' | 'manager' | 'expert';
}

// Disease detection types
export interface DetectionResult {
  class: string;
  confidence: number;
  severity: 'None' | 'Low' | 'Medium' | 'High';
  treatment: string;
  timestamp?: string;
}

export interface DiseaseInfo {
  description: string;
  symptoms: string[];
  causes: string[];
  treatment: string[];
  prevention: string[];
  severity: 'None' | 'Low' | 'Medium' | 'High';
  prevalence: string;
  color: string;
}

// History types
export interface HistoryItem {
  id: string;
  date: string;
  time: string;
  image: string;
  result: string;
  confidence: number;
  severity: 'None' | 'Low' | 'Medium' | 'High';
  treatment: string;
  location?: string;
  timestamp: string;
}

// API types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface DetectionApiResponse {
  filename: string;
  timestamp: string;
  predictions: Array<{
    class: string;
    confidence: number;
    class_id: number;
    timestamp: string;
  }>;
  detailed_predictions: Array<{
    class: string;
    confidence: number;
    class_id: number;
    timestamp: string;
    disease_info: DiseaseInfo;
  }>;
  total_detections: number;
  health_status: string;
  severity: string;
  recommendations: string[];
}

// Navigation types
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
  Home: undefined;
  Camera: undefined;
  History: undefined;
  Info: undefined;
  UploadImage: undefined;
  DetectionResult: {
    image: string;
    result: DetectionResult;
    fromHistory?: boolean;
  };
};

// Form types
export interface LoginForm {
  email: string;
  password: string;
}

export interface RegisterForm {
  name: string;
  email: string;
  password: string;
  farmName?: string;
}

// Statistics types
export interface FarmStats {
  totalScans: number;
  healthyPlants: number;
  diseasedPlants: number;
  accuracyRate: number;
  lastScanDate?: string;
}

// Upload types
export interface UploadedImage {
  uri: string;
  name: string;
  size: number;
  type: string;
}

// Settings types
export interface AppSettings {
  notifications: boolean;
  autoSave: boolean;
  confidenceThreshold: number;
  language: string;
  theme: 'light' | 'dark' | 'auto';
}

// Error types
export interface AppError {
  code: string;
  message: string;
  details?: any;
}