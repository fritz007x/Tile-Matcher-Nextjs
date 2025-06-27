// User related types
export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
  updatedAt: string;
}

// Auth related types
export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  name: string;
  email: string;
  password: string;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  password: string;
}

// Tile matching related types
export interface MatchResultItem {
  tile_id: string;
  score: number;
  method: string;
  metadata: {
    sku: string;
    model_name: string;
    collection_name: string;
    image_url: string;
    [key: string]: any;
  };
}

export interface MatchResult {
  id: string;
  createdAt: string;
  queryImage: string;
  results: MatchResultItem[];
}

// API response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}

export interface ApiError {
  message: string;
  status: number;
  errors?: Record<string, string[]>;
}

// Component props
export interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  text?: string;
  fullScreen?: boolean;
}
