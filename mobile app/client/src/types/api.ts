export interface DiseaseInfo {
  description: string;
  symptoms: string[];
  treatment: string;
  severity: 'None' | 'Low' | 'Medium' | 'High';
  color: string;
  prevention?: string[];
}

export interface Prediction {
  class: string;
  confidence: number;
  class_id: number;
  timestamp: string;
  disease_info?: DiseaseInfo;
}

export interface DetectionResult {
  filename: string;
  timestamp: string;
  predictions: Prediction[];
  detailed_predictions: Prediction[];
  total_detections: number;
  image_size: {
    width: number;
    height: number;
  };
  health_status: string;
  severity: string;
  recommendations: string[];
}

export interface BatchDetectionResult {
  batch_results: Array<{
    index: number;
    filename: string;
    predictions: Prediction[];
    timestamp: string;
    error?: string;
  }>;
  total_processed: number;
  timestamp: string;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface HealthCheckResponse {
  status: string;
  timestamp: string;
  model_status: string;
}

export interface DiseasesResponse {
  diseases: Record<string, DiseaseInfo>;
  total_diseases: number;
}

export interface DetectionHistory {
  id: string;
  imageUri: string;
  result: DetectionResult;
  timestamp: string;
}