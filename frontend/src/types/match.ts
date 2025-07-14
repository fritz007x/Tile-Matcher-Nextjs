export interface MatchResultItem {
  id?: string;  // Optional since it might come from different sources
  tile_id: string;
  imageUrl?: string;  // Optional since we might have image_data
  similarity: number;
  score: number;
  method: string;
  // New fields for Base64 image support
  image_data?: string;    // Base64-encoded image data
  content_type?: string;  // MIME type of the image
  has_image_data?: boolean; // Flag to indicate if image_data is available
  metadata: {
    model_name?: string;
    collection_name?: string;
    sku?: string;
    image_url?: string;
    description?: string;
    [key: string]: any;
  };
}

export interface BackendMatchItem {
  id: string;
  sku: string;
  model_name: string;
  collection_name: string;
  image_path?: string;
  created_at: string;
  updated_at: string;
  description: string | null;
  image_data?: string;
  content_type?: string;
  has_image_data?: boolean;
}

export interface BackendMatchResponse {
  query_filename: string;
  matches: BackendMatchItem[];
  scores: number[];
}

