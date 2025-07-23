import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { ApiResponse, ApiError, AuthToken, LoginCredentials, RegisterData, PasswordResetRequest, PasswordResetConfirm } from '@/types';
import { MatchResultItem, BackendMatchResponse } from '@/types/match';
import { APP_CONFIG } from '@/config';
import AuthService from './auth';

// Create axios instance with base configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: APP_CONFIG.apiBaseUrl,
  // headers: {
  //   'Content-Type': 'application/json',
  // },
  timeout: 30000, // 30 seconds
});

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = AuthService.getToken();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Only set Content-Type to application/json if it's not FormData
    if (!(config.data instanceof FormData)) {
      config.headers['Content-Type'] = 'application/json';
    }
    // For FormData, let axios handle Content-Type automatically
    
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor to handle errors
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    // Handle token expiration
    if (error.response?.status === 401) {
      AuthService.signOut();
      window.location.href = '/login?session=expired';
    }
    return Promise.reject(error);
  }
);

// Wrapper function to standardize API responses
const apiRequest = async <T>(config: AxiosRequestConfig): Promise<ApiResponse<T>> => {
  try {
    const response: AxiosResponse = await apiClient(config);
    return {
      data: response.data,
      status: response.status,
      message: response.statusText
    };
  } catch (error: any) {
    // Handle new structured error format from backend
    const errorData = error.response?.data;
    
    const apiError: ApiError = {
      message: errorData?.message || errorData?.detail?.message || errorData?.detail || error.message || 'An error occurred',
      status: error.response?.status || 500,
      errors: errorData?.errors || errorData?.detail?.errors,
      errorCode: errorData?.error_code || errorData?.detail?.error_code,
      details: errorData?.details || errorData?.detail?.details
    };
    throw apiError;
  }
};

// Auth related API endpoints
const auth = {
  login: (credentials: LoginCredentials) => 
    apiRequest<AuthToken>({
      method: 'POST',
      url: '/token',
      data: credentials
    }),
    
  register: (userData: RegisterData) => 
    apiRequest<{ user: any }>({
      method: 'POST',
      url: '/api/auth/register',
      data: userData
    }),
    
  forgotPassword: (data: PasswordResetRequest) => 
    apiRequest<{ message: string }>({
      method: 'POST',
      url: '/api/auth/forgot-password',
      data
    }),
    
  resetPassword: (data: PasswordResetConfirm) => 
    apiRequest<{ message: string }>({
      method: 'POST',
      url: '/api/auth/reset-password',
      data
    }),
    
  me: () => 
    apiRequest<{ user: any }>({
      method: 'GET',
      url: '/api/auth/me'
    })
};

// Tile matching related API endpoints
const matching = {
  match: (formData: FormData) => 
    apiRequest<BackendMatchResponse>({
      method: 'POST',
      url: '/api/matching/match',
      data: formData,
    }),
    
  getHistory: () => 
    apiRequest<MatchResultItem[]>({
      method: 'GET',
      url: '/api/matching/history'
    }),
    
  getMatchById: (id: string) => 
    apiRequest<MatchResultItem>({
      method: 'GET',
      url: `/api/matching/history/${id}`
    }),
    
  saveMatch: (id: string) => 
    apiRequest<{ message: string }>({
      method: 'POST',
      url: `/api/matching/${id}/save`
    }),
    
  deleteMatch: (id: string) => 
    apiRequest<{ message: string }>({
      method: 'DELETE',
      url: `/api/matching/${id}`
    }),
    
  getThumbnailUrl: (tileId: string, width: number = 200, height: number = 200) => 
    `${APP_CONFIG.apiBaseUrl}/api/matching/tile/${tileId}/thumbnail?width=${width}&height=${height}`
};

// User profile related API endpoints
const user = {
  updateProfile: (userData: { name: string, email: string }) => 
    apiRequest<{ user: any }>({
      method: 'PUT',
      url: '/api/user/profile',
      data: userData
    }),
    
  changePassword: (data: { currentPassword: string, newPassword: string }) => 
    apiRequest<{ message: string }>({
      method: 'PUT',
      url: '/api/user/change-password',
      data
    }),
    
  deleteAccount: () => 
    apiRequest<{ message: string }>({
      method: 'DELETE',
      url: '/api/user/account'
    }),
    
  getSavedMatches: () => 
    apiRequest<MatchResultItem[]>({
      method: 'GET',
      url: '/api/user/saved-matches'
    }),
};

export const apiService = {
  auth,
  matching,
  user
};
