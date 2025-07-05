'use client';

import { useState } from 'react';
import Footer from '@/components/Footer';
import ImageUpload from '@/components/ImageUpload';
import MatchResults from '@/components/MatchResults';
import { apiService } from '@/lib/api';
import { MatchResultItem } from '@/types/match';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matchResults, setMatchResults] = useState<MatchResultItem[]>([]);
  
  const handleImageUpload = async (file: File) => {
    if (!file) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      // Call the API to match the uploaded image
      const formData = new FormData();
      formData.append('image', file);
      const response = await apiService.matching.match(formData);
      // The API returns an array of MatchResultItem
      if (!response.data) {
        throw new Error('No data received from the server');
      }
      
      // Ensure we have an array
      const results = Array.isArray(response.data) ? response.data : [response.data];
      
      // Transform the data if needed to match our frontend type
      const formattedResults = results.map(item => ({
        ...item,
        // Add any necessary transformations here
        // For example, ensure imageUrl is set correctly
        imageUrl: item.imageUrl || item.metadata?.image_url || '',
        // Ensure similarity is a number
        similarity: typeof item.similarity === 'number' ? item.similarity : (item.score || 0) * 100,
        // Ensure metadata exists
        metadata: item.metadata || {}
      }));
      
      setMatchResults(formattedResults);
    } catch (err) {
      console.error('Error matching image:', err);
      setError('Failed to match the image. Please try again.');
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
          
          {matchResults.length > 0 && <MatchResults results={matchResults} />}
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
