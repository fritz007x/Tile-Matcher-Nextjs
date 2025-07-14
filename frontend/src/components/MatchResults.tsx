import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { MatchResultItem } from '@/types/match';
import { APP_CONFIG } from '@/config';
import LoadingSpinner, { LoadingCard } from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

// Helper function to get image source URL
const getImageSource = (result: MatchResultItem) => {
  // If we have base64 data, use it directly
  if ('image_data' in result && result.image_data) {
    const dataUri = `data:${result.content_type || 'image/jpeg'};base64,${result.image_data}`;
    return dataUri;
  }
  
  // Fallback to imageUrl if available
  if (result.imageUrl) {
    return result.imageUrl;
  }
  
  // Return null if no image source is available
  return null;
};

interface MatchResultsProps {
  results: MatchResultItem[];
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

const MatchResults: React.FC<MatchResultsProps> = ({ results = [], isLoading = false, error = null, onRetry }) => {
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());

  const handleImageError = (imageKey: string, event: any) => {
    console.error(`Image failed to load for key ${imageKey}:`, event);
    setFailedImages(prev => new Set(prev).add(imageKey));
  };

  // Reset failed images when results change
  useEffect(() => {
    setFailedImages(new Set());
  }, [results]);

  if (error) {
    return (
      <div className="mt-8">
        <ErrorMessage
          error={error}
          onRetry={onRetry}
          variant="card"
          retryText="Try Again"
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="mt-8">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Processing your image...</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="border rounded-lg overflow-hidden shadow-sm">
              <LoadingCard className="h-48" />
              <div className="p-4 space-y-2">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="mt-8 text-center py-8">
        <div className="mb-4">
          <svg 
            className="w-12 h-12 text-gray-400 mx-auto" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" 
            />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No matches found</h3>
        <p className="text-gray-600">
          Try adjusting your search criteria or uploading a different image.
        </p>
      </div>
    );
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
              <div className="aspect-w-1 aspect-h-1 w-full overflow-hidden rounded-lg bg-gray-100 relative">
                {(() => {
                  const imageSource = getImageSource(result);
                  const imageKey = `${result.tile_id}-${index}`;

                  if (imageSource && !failedImages.has(imageKey)) {
                    // Use regular img tag for base64 data URIs (Next.js Image component doesn't handle them well)
                    if (imageSource.startsWith('data:')) {
                      return (
                        <img
                          src={imageSource}
                          alt={`${result.metadata?.model_name || 'Tile'} image`}
                          className="w-full h-full object-cover object-center"
                          onError={(e) => handleImageError(imageKey, e)}
                          style={{ aspectRatio: '1 / 1' }}
                        />
                      );
                    } else {
                      // Use Next.js Image component for regular URLs
                      return (
                        <Image
                          src={imageSource}
                          alt={`${result.metadata?.model_name || 'Tile'} image`}
                          fill
                          className="object-cover object-center"
                          onError={(e) => handleImageError(imageKey, e)}
                          placeholder="blur"
                          blurDataURL="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
                        />
                      );
                    }
                  } else {
                    return (
                      <div className="flex h-full items-center justify-center bg-gray-200">
                        <span className="text-gray-500">Image not available</span>
                      </div>
                    );
                  }
                })()}
              </div>
              
              <div className="mt-4">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-gray-900 truncate text-base">
                      {result.metadata?.model_name || 'Unknown Tile'}
                    </h4>
                    <p className="text-sm text-gray-600 mt-1">
                      {result.metadata?.collection_name || 'Unknown Collection'}
                    </p>
                  </div>
                  <div className="ml-3 flex-shrink-0">
                    <div className="px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 whitespace-nowrap">
                      {(result.similarity ?? result.score ?? 0).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <dl className="space-y-2 text-sm">
                  {result.metadata?.sku && result.metadata.sku !== 'N/A' && (
                    <div className="flex justify-between">
                      <dt className="font-medium text-gray-700">SKU:</dt>
                      <dd className="text-gray-900 font-mono text-xs">{result.metadata.sku}</dd>
                    </div>
                  )}
                  
                  <div className="flex justify-between">
                    <dt className="font-medium text-gray-700">Method:</dt>
                    <dd className="text-gray-900 capitalize">{result.method.replace('_', ' ')}</dd>
                  </div>
                  
                  <div className="flex justify-between">
                    <dt className="font-medium text-gray-700">Tile ID:</dt>
                    <dd className="text-gray-900 font-mono text-xs">{result.tile_id}</dd>
                  </div>
                  
                  {result.metadata?.description && (
                    <div>
                      <dt className="font-medium text-gray-700 mb-1">Description:</dt>
                      <dd className="text-gray-900 text-xs leading-relaxed">{result.metadata.description}</dd>
                    </div>
                  )}
                </dl>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MatchResults;