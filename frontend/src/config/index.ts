interface AppConfig {
  apiBaseUrl: string;
  imageUploadSizeLimit: number; // in MB
  appName: string;
  contactEmail: string;
  maxResultsToShow: number;
  defaultMatchThreshold: number; // 0-1 score threshold
  supportedImageFormats: string[];
}

// Default configuration (development)
const defaultConfig: AppConfig = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  imageUploadSizeLimit: 5, // 5MB
  appName: 'Tile Matcher',
  contactEmail: 'support@tilematcher.com',
  maxResultsToShow: 10,
  defaultMatchThreshold: 0.5,
  supportedImageFormats: ['jpg', 'jpeg', 'png', 'webp']
};

// Production overrides
const productionConfig: Partial<AppConfig> = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_URL || 'https://api.tilematcher.com',
  // Any other production-specific settings
};

// Configuration based on environment
const config: AppConfig = {
  ...defaultConfig,
  ...(process.env.NODE_ENV === 'production' ? productionConfig : {})
};

export default config;
export const APP_CONFIG = config;
