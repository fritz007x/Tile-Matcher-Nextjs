import React from 'react';
import Image from 'next/image';
import { MatchResultItem } from '@/lib/api';

interface MatchResultsProps {
  results: MatchResultItem[];
  isLoading?: boolean;
}

const MatchResults: React.FC<MatchResultsProps> = ({ results, isLoading = false }) => {
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
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="font-medium text-gray-900 truncate">
                    {result.metadata?.model_name || 'Unknown Tile'}
                  </h4>
                  <p className="text-sm text-gray-500">
                    {result.metadata?.collection_name || 'Unknown Collection'}
                  </p>
                </div>
                <div className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  {Math.round(result.score * 100)}% match
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
