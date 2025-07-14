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
            image_url: match.image_path,
            description: match.description || undefined,
            created_at: match.created_at,
            updated_at: match.updated_at
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
  
  // Available matching methods - These should match what the backend supports
  const availableMethods = [
    { id: 'color_hist', name: 'Color Histogram' },
    { id: 'orb', name: 'ORB Features' },
    { id: 'vit_simple', name: 'Vision Transformer (Simple)' },
    { id: 'vit_multi_layer', name: 'Vision Transformer (Multi-Layer)' },
    { id: 'vit_multi_scale', name: 'Vision Transformer (Multi-Scale)' },
    { id: 'ensemble', name: 'Ensemble (Multiple Methods)' }
  ];
  
  // Available top_k options
  const topKOptions = [1, 3, 5, 10, 15, 20];
  
  return (
    <main className="flex min-h-screen flex-col">
      
      <div className="container mx-auto px-4 flex-grow py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-center mb-4">Tile Matcher</h1>
          <p className="text-center text-lg text-gray-600 mb-12">
            Upload an image of a tile to find the best matches in our catalog
          </p>
          
          {/* Matching Options Panel */}
          <div className="mb-8 p-4 bg-gray-50 rounded-lg shadow-sm">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Matching Options</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Method Selection */}
              <div>
                <label htmlFor="method-select" className="block text-sm font-medium text-gray-700 mb-1">
                  Matching Method
                </label>
                <select
                  id="method-select"
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  {availableMethods.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {method === 'color_hist' && "Compares color distributions between images."}
                  {method === 'orb' && "Detects key points and features in the images."}
                  {method === 'vit_simple' && "Simple Vision Transformer for basic visual pattern recognition."}
                  {method === 'vit_multi_layer' && "Multi-layer Vision Transformer for detailed feature extraction."}
                  {method === 'vit_multi_scale' && "Multi-scale Vision Transformer for robust pattern matching."}
                  {method === 'ensemble' && "Combines multiple methods for best accuracy."}
                </p>
              </div>
              
              {/* Top K Selection */}
              <div>
                <label htmlFor="topk-select" className="block text-sm font-medium text-gray-700 mb-1">
                  Number of Results (Top K): {topK}
                </label>
                <select
                  id="topk-select"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  {topKOptions.map(k => (
                    <option key={k} value={k}>{k}</option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  Maximum number of similar tiles to return.
                </p>
              </div>
              
              {/* Threshold Slider */}
              <div className="md:col-span-2">
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
                <p className="mt-1 text-xs text-gray-500">
                  Only show results with similarity above this threshold.
                </p>
              </div>
            </div>
          </div>
          
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
          
          {matchResults.length > 0 && <MatchResults results={matchResults} />}
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
