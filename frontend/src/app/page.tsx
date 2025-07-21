'use client';

import { useState } from 'react';
import Footer from '@/components/Footer';
import ImageUpload from '@/components/ImageUpload';
import MatchResults from '@/components/MatchResults';
import { apiService } from '@/lib/api';
import { MatchResultItem, BackendMatchResponse, BackendMatchItem } from '@/types/match';
import { APP_CONFIG } from '@/config';



export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matchResults, setMatchResults] = useState<MatchResultItem[]>([]);

  // Set default values for matching parameters
  const [topK, setTopK] = useState<number>(3);
  const [threshold, setThreshold] = useState<number>(70); // 70% threshold by default
  const [method, setMethod] = useState<string>('enhanced_vit_multi_layer');

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
        const imageUrl = apiService.matching.getThumbnailUrl(match.id);

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
          },
          imageUrl: imageUrl
        };

        return result;
      });

      console.log('Formatted results:', formattedResults);
      setMatchResults(formattedResults);
    } catch (err) {
      console.error('Error matching image:', err);
      console.error('Error details:', {
        message: err instanceof Error ? err.message : 'Unknown error',
        stack: err instanceof Error ? err.stack : undefined,
        response: err?.response?.data || 'No response data'
      });
      setError(err instanceof Error ? err.message : 'Failed to match the image. Please try again.');
      setMatchResults([]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Available matching methods - These should match what the backend supports
  const availableMethods = [
    { id: 'color_hist', name: 'Color Histogram', category: 'Traditional', speed: 'Fast', accuracy: 'Basic' },
    { id: 'orb', name: 'ORB Features', category: 'Traditional', speed: 'Fast', accuracy: 'Good' },
    { id: 'vit_simple', name: 'Vision Transformer (Simple)', category: 'AI - Basic', speed: 'Medium', accuracy: 'Good' },
    { id: 'vit_multi_layer', name: 'Vision Transformer (Multi-Layer)', category: 'AI - Basic', speed: 'Medium', accuracy: 'Very Good' },
    { id: 'vit_multi_scale', name: 'Vision Transformer (Multi-Scale)', category: 'AI - Basic', speed: 'Slow', accuracy: 'Very Good' },
    { id: 'enhanced_vit_single', name: 'Enhanced ViT (Single-Layer)', category: 'AI - Enhanced', speed: 'Medium', accuracy: 'Excellent', recommended: true },
    { id: 'enhanced_vit_multi_layer', name: 'Enhanced ViT (Multi-Layer) ⭐', category: 'AI - Enhanced', speed: 'Medium', accuracy: 'Excellent', recommended: true },
    { id: 'enhanced_vit_multi_scale', name: 'Enhanced ViT (Multi-Scale)', category: 'AI - Enhanced', speed: 'Slow', accuracy: 'Best', recommended: true },
    { id: 'ensemble', name: 'Ensemble (Traditional + Enhanced ViT)', category: 'Ensemble', speed: 'Slow', accuracy: 'Excellent' },
    { id: 'enhanced_ensemble', name: 'Enhanced Ensemble (All Methods) 🏆', category: 'Ensemble', speed: 'Slowest', accuracy: 'Best', recommended: true },
    { id: 'advanced_preprocessing', name: 'Advanced Preprocessing + Traditional', category: 'Specialized', speed: 'Medium', accuracy: 'Good' }
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
                  {/* Group methods by category */}
                  {['Traditional', 'AI - Basic', 'AI - Enhanced', 'Ensemble', 'Specialized'].map(category => {
                    const categoryMethods = availableMethods.filter(m => m.category === category);
                    if (categoryMethods.length === 0) return null;
                    
                    return (
                      <optgroup key={category} label={category}>
                        {categoryMethods.map(m => (
                          <option key={m.id} value={m.id}>{m.name}</option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {method === 'color_hist' && "Compares color distributions between images."}
                  {method === 'orb' && "Detects key points and features in the images."}
                  {method === 'vit_simple' && "Simple Vision Transformer for basic visual pattern recognition."}
                  {method === 'vit_multi_layer' && "Multi-layer Vision Transformer for detailed feature extraction."}
                  {method === 'vit_multi_scale' && "Multi-scale Vision Transformer for robust pattern matching."}
                  {method === 'enhanced_vit_single' && "Enhanced ViT with improved single-layer feature extraction and patch aggregation."}
                  {method === 'enhanced_vit_multi_layer' && "Enhanced ViT combining features from multiple transformer layers for richer representation."}
                  {method === 'enhanced_vit_multi_scale' && "Enhanced ViT with multi-layer features at multiple image scales for maximum accuracy."}
                  {method === 'ensemble' && "Combines traditional methods with Enhanced ViT for balanced accuracy."}
                  {method === 'enhanced_ensemble' && "Advanced ensemble using all Enhanced ViT variants for ultimate precision."}
                  {method === 'advanced_preprocessing' && "Uses advanced image preprocessing with lighting normalization and tile extraction."}
                </p>
                
                {/* Performance indicators */}
                {(() => {
                  const selectedMethod = availableMethods.find(m => m.id === method);
                  if (!selectedMethod) return null;
                  
                  return (
                    <div className="mt-2 flex gap-4 text-xs">
                      <div className="flex items-center gap-1">
                        <span className="font-medium text-gray-600">Speed:</span>
                        <span className={`px-2 py-1 rounded-full text-white text-xs ${
                          selectedMethod.speed === 'Fast' ? 'bg-green-500' :
                          selectedMethod.speed === 'Medium' ? 'bg-yellow-500' :
                          selectedMethod.speed === 'Slow' ? 'bg-orange-500' : 'bg-red-500'
                        }`}>
                          {selectedMethod.speed}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="font-medium text-gray-600">Accuracy:</span>
                        <span className={`px-2 py-1 rounded-full text-white text-xs ${
                          selectedMethod.accuracy === 'Basic' ? 'bg-gray-500' :
                          selectedMethod.accuracy === 'Good' ? 'bg-blue-500' :
                          selectedMethod.accuracy === 'Very Good' ? 'bg-indigo-500' :
                          selectedMethod.accuracy === 'Excellent' ? 'bg-purple-500' : 'bg-pink-500'
                        }`}>
                          {selectedMethod.accuracy}
                        </span>
                      </div>
                      {selectedMethod.recommended && (
                        <div className="flex items-center gap-1">
                          <span className="px-2 py-1 rounded-full bg-green-100 text-green-800 text-xs font-medium">
                            Recommended
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })()}
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
