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
export interface MatchResult {
  query_id: string;
  timestamp: string;
  results: import('./match').MatchResultItem[];
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
  errorCode?: string;
  details?: Record<string, any>;
}

// Component props
export interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  text?: string;
  fullScreen?: boolean;
}

// Tile and catalog related types
export interface TileSearchFilters {
  sku: string;
  modelName: string;
  collectionName: string;
  description: string;
  createdAfter: string;
  limit: number;
  offset: number;
}

export interface TileResult {
  id: string;
  sku: string;
  model_name: string;
  collection_name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  has_image_data: boolean;
  image_data?: string;
  content_type: string;
}

export interface TileSearchResponse {
  results: TileResult[];
  total: number;
  limit: number;
  offset: number;
}

export interface TileSearchProps {
  onTileSelect?: (tile: TileResult) => void;
  maxResults?: number;
  showImages?: boolean;
  allowMultiSelect?: boolean;
  className?: string;
}
