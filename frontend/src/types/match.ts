export interface MatchResultItem {
  id?: string;  // Optional since it might come from different sources
  tile_id: string;
  imageUrl: string;
  similarity: number;
  score: number;
  method: string;
  metadata: {
    model_name?: string;
    collection_name?: string;
    sku?: string;
    image_url?: string;
    [key: string]: any;
  };
}

