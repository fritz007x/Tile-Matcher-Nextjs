export interface MatchResultItem {
  id: string;
  tile_id: string;
  imageUrl: string;
  similarity: number;
  score: number;
  method: string;
  metadata?: {
    model_name?: string;
    collection_name?: string;
    sku?: string;
    [key: string]: any;
  };
}
