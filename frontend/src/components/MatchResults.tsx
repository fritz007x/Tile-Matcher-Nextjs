import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { MatchResultItem } from '@/types/match';

// Helper function to get image source URL
const getImageSource = (result: MatchResultItem) => {
  // If we have base64 data, use it directly
  if ('image_data' in result && result.image_data) {
    return `data:${result.content_type || 'image/jpeg'};base64,${result.image_data}`;
  }
  
  // Fall back to imageUrl if available
  if (result.imageUrl) {
    return result.imageUrl;
  }
  
  // Fallback to a placeholder if no image is available
  return '/placeholder-image.jpg';
};

interface MatchResultsProps {
  results: MatchResultItem[];
  isLoading?: boolean;
  error?: string | null;
}

const MatchResults: React.FC<MatchResultsProps> = ({ results = [], isLoading = false, error = null }) => {
  if (isLoading) {
    return (
      <div className="mt-8 space-y-4">
        <h3 className="text-lg font-medium text-gray-900">Processing your image...</h3>
        <div className="flex justify-center">
          <div className="animate-pulse h-4 bg-gray-300 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return null;
  }

  return (
    <div className="mt-8">
      <h3 className="text-lg font-medium text-gray-900">Match Results</h3>
      <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {results.map((result, index) => (
          <div
            key={`${result.tile_id}-${index}`}
            className="border rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="p-4">
              <div className="aspect-w-1 aspect-h-1 w-full overflow-hidden rounded-lg bg-gray-100">
                {result.has_image_data || result.imageUrl ? (
                  <img
                    src={getImageSource(result)}
                    alt={`${result.metadata?.model_name || 'Tile'} image`}
                    className="h-full w-full object-cover object-center"
                    onError={(e) => {
                      // Fallback to placeholder if image fails to load
                      const target = e.target as HTMLImageElement;
                      target.src = '/placeholder-image.jpg';
                    }}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center bg-gray-200">
                    <span className="text-gray-500">No image available</span>
                  </div>
                )}
              </div>
              
              <div className="mt-4 flex justify-between items-start">
                <div>
                  <h4 className="font-medium text-gray-900 truncate">
                    {result.metadata?.model_name || 'Unknown Tile'}
                  </h4>
                  <p className="text-sm text-gray-500">
                    {result.metadata?.collection_name || 'Unknown Collection'}
                  </p>
                </div>
                <div className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 whitespace-nowrap">
                  {Math.round(result.similarity || result.score * 100)}% match
                </div>
              </div>

              <dl className="mt-2 text-sm text-gray-500">
                {result.metadata?.sku && (
                  <>
                    <dt className="inline font-medium text-gray-700">SKU: </dt>
                    <dd className="inline ml-1">{result.metadata.sku}</dd>
                  </>
                )}
                
                <div className="mt-1">
                  <dt className="inline font-medium text-gray-700">Method: </dt>
                  <dd className="inline ml-1">{result.method}</dd>
                </div>
              </dl>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MatchResults;
