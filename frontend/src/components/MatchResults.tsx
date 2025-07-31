import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { MatchResultItem } from '@/types/match';
import { APP_CONFIG } from '@/config';
import LoadingSpinner, { LoadingCard } from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

// Helper function to get image source URL
const getImageSource = (result: MatchResultItem) => {
  return result.imageUrl || null;
};

interface MatchResultsProps {
  results: MatchResultItem[];
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

const MatchResults: React.FC<MatchResultsProps> = ({ results = [], isLoading = false, error = null, onRetry }) => {
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const [loadingImages, setLoadingImages] = useState<Set<string>>(new Set());

  const handleImageError = (imageKey: string, event: any) => {
    console.error(`Image failed to load for key ${imageKey}:`, event);
    setFailedImages(prev => new Set(prev).add(imageKey));
    setLoadingImages(prev => {
      const newSet = new Set(prev);
      newSet.delete(imageKey);
      return newSet;
    });
  };

  const handleImageLoadStart = (imageKey: string) => {
    setLoadingImages(prev => new Set(prev).add(imageKey));
  };

  const handleImageLoadComplete = (imageKey: string) => {
    setLoadingImages(prev => {
      const newSet = new Set(prev);
      newSet.delete(imageKey);
      return newSet;
    });
  };

  // Reset failed and loading images when results change
  useEffect(() => {
    setFailedImages(new Set());
    setLoadingImages(new Set());
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
              {/* Image placeholder with loading animation */}
              <div className="aspect-square w-full bg-gray-100 relative">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="animate-pulse">
                    <svg 
                      className="w-12 h-12 text-gray-300" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={1.5} 
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" 
                      />
                    </svg>
                  </div>
                </div>
                {/* Score placeholder */}
                <div className="absolute top-2 right-2">
                  <div className="w-12 h-6 bg-gray-300 rounded animate-pulse"></div>
                </div>
              </div>
              {/* Content placeholders */}
              <div className="p-4 space-y-2">
                <div className="h-4 bg-gray-200 rounded w-3/4 animate-pulse"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 animate-pulse"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3 animate-pulse"></div>
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
            {/* Image Thumbnail Section */}
            <div className="relative">
              <div className="aspect-square w-full overflow-hidden bg-gray-100 relative">
                {(() => {
                  const imageSource = getImageSource(result);
                  const imageKey = `${result.tile_id}-${index}`;

                  if (imageSource && !failedImages.has(imageKey)) {
                    return (
                      <div className="relative h-full">
                        <Image
                          src={imageSource}
                          alt={`${result.metadata?.model_name || 'Tile'} image`}
                          fill
                          className="object-cover object-center"
                          onError={(e) => handleImageError(imageKey, e)}
                          onLoadStart={() => handleImageLoadStart(imageKey)}
                          onLoad={() => handleImageLoadComplete(imageKey)}
                          placeholder="blur"
                          blurDataURL="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
                        />
                        {loadingImages.has(imageKey) && (
                          <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
                            <div className="animate-pulse">
                              <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  } else {
                    return (
                      <div className="flex h-full items-center justify-center bg-gray-200">
                        <div className="text-center p-4">
                          <svg 
                            className="w-12 h-12 text-gray-400 mx-auto mb-2" 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                          >
                            <path 
                              strokeLinecap="round" 
                              strokeLinejoin="round" 
                              strokeWidth={1.5} 
                              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" 
                            />
                          </svg>
                          <span className="text-sm text-gray-500">No image</span>
                        </div>
                      </div>
                    );
                  }
                })()}
              </div>
              
              {/* Similarity Score Badge */}
              <div className="absolute top-2 right-2">
                <div className="px-2 py-1 rounded-md text-xs font-medium bg-black bg-opacity-75 text-white">
                  {(result.similarity ?? result.score ?? 0).toFixed(1)}%
                </div>
              </div>
            </div>
            
            {/* Content Section */}
            <div className="p-4">
              <div className="mb-3">
                <h4 className="font-medium text-gray-900 truncate text-base">
                  {result.metadata?.model_name || 'Unknown Tile'}
                </h4>
                <p className="text-sm text-gray-600 mt-1">
                  {result.metadata?.collection_name || 'Unknown Collection'}
                </p>
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
                
                
                {result.metadata?.description && (
                  <div>
                    <dt className="font-medium text-gray-700 mb-1">Description:</dt>
                    <dd className="text-gray-900 text-xs leading-relaxed">{result.metadata.description}</dd>
                  </div>
                )}
              </dl>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MatchResults;