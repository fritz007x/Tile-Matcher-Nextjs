'use client';

import { useState } from 'react';
import Footer from '@/components/Footer';
import ImageUpload from '@/components/ImageUpload';
import MatchResults from '@/components/MatchResults';
import { apiService } from '@/lib/api';
import { MatchResultItem, BackendMatchResponse, BackendMatchItem } from '@/types/match';



export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matchResults, setMatchResults] = useState<MatchResultItem[]>([]);

  // Set default values for matching parameters
  const [topK, setTopK] = useState<number>(3);
  const [threshold, setThreshold] = useState<number>(70); // 70% threshold by default
  const [method, setMethod] = useState<string>('color_hist');

  const handleImageUpload = async (file: File) => {
    if (!file) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('top_k', topK.toString());
      formData.append('threshold', (threshold / 100).toString()); // Convert % to 0-1 range
      formData.append('method', method);

      console.log('Sending request with method:', method, 'threshold:', threshold / 100);
      
      const response = await apiService.matching.match(formData);
      const backendData = response.data as unknown as BackendMatchResponse;
      
      console.log('Backend response:', backendData);
      
      if (!backendData || !backendData.matches || !Array.isArray(backendData.matches) || !Array.isArray(backendData.scores)) {
        console.error('Invalid response format:', backendData);
        throw new Error('Invalid response format from server');
      }

      const formattedResults: MatchResultItem[] = backendData.matches.map((match: BackendMatchItem, index: number) => {
        // Check if we have Base64 image data from the backend
        const hasImageData = match.has_image_data && match.image_data;
        
        // Only create a URL if we don't have base64 data
        const imageUrl = !hasImageData && match.image_path
          ? match.image_path.startsWith('http')
            ? match.image_path 
            : `/api/images?path=${encodeURIComponent(match.image_path)}`
          : undefined;
          
        const result: MatchResultItem = {
          tile_id: match.id,
          id: match.id,
          similarity: Math.round(backendData.scores[index] * 10000) / 100, // Convert to percentage with 2 decimal places
          score: backendData.scores[index],
          method: method || 'color_hist',
          metadata: {
            sku: match.sku || 'N/A',
            model_name: match.model_name || 'Unknown',
            collection_name: match.collection_name || 'Unknown',
            image_url: match.image_path
          }
        };
        
        // Add image data if available
        if (hasImageData && match.image_data) {
          result.image_data = match.image_data;
          result.content_type = match.content_type || 'image/jpeg';
          result.has_image_data = true;
        } else if (imageUrl) {
          result.imageUrl = imageUrl;
        }
        
        return result;
      });

      console.log('Formatted results:', formattedResults);
      setMatchResults(formattedResults);
    } catch (err) {
      console.error('Error matching image:', err);
      setError(err instanceof Error ? err.message : 'Failed to match the image. Please try again.');
      setMatchResults([]);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <main className="flex min-h-screen flex-col">
      
      <div className="container mx-auto px-4 flex-grow py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-center mb-4">Tile Matcher</h1>
          <p className="text-center text-lg text-gray-600 mb-12">
            Upload an image of a tile to find the best matches in our catalog
          </p>
          
          <ImageUpload onImageUploaded={handleImageUpload} isDisabled={isLoading} />
          
          {isLoading && (
            <div className="text-center mt-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
              <p className="mt-2 text-gray-600">Analyzing your image...</p>
              <p className="text-sm text-gray-500">This may take up to 10 seconds</p>
            </div>
          )}
          
          {error && (
            <div className="mt-8 p-4 bg-red-50 text-red-700 rounded-md">
              {error}
            </div>
          )}
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Similarity Threshold: {threshold}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={threshold}
              onChange={(e) => setThreshold(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
          
          {matchResults.length > 0 && <MatchResults results={matchResults} />}
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
