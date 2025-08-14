'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import Image from 'next/image';
import { LoadingSpinner } from './LoadingSpinner';
import { ErrorMessage } from './ErrorMessage';
import { 
  TileSearchFilters, 
  TileResult, 
  TileSearchResponse, 
  TileSearchProps 
} from '@/types';

// Component to fetch and display tile thumbnail
function TileThumbnail({ tileId, sku, className = '' }: { tileId: string; sku: string; className?: string }) {
  const { data: session } = useSession();
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchThumbnail = async () => {
      try {
        console.log(`🖼️ Fetching thumbnail for tile ${tileId}`);
        const headersObj: Record<string, string> = {};
        if (session?.user?.accessToken) {
          headersObj['Authorization'] = `Bearer ${session.user.accessToken}`;
        }

        // Use Next.js API proxy route to avoid browser security restrictions
        const response = await fetch(`/api/tile-thumbnail/${tileId}?width=200&height=200`, {
          headers: headersObj,
        });

        console.log(`🖼️ Thumbnail response for ${tileId}:`, response.status);

        if (!response.ok) {
          const errorText = await response.text();
          console.error(`❌ Thumbnail fetch failed for ${tileId}:`, errorText);
          throw new Error(`Failed to fetch thumbnail: ${response.status}`);
        }

        const data = await response.json();
        console.log(`✅ Thumbnail data received for ${tileId}:`, { 
          content_type: data.content_type, 
          has_data: !!data.data,
          data_length: data.data?.length 
        });
        setImageSrc(`data:${data.content_type};base64,${data.data}`);
      } catch (err) {
        console.error(`❌ Thumbnail error for ${tileId}:`, err);
        setError(true);
      } finally {
        setIsLoading(false);
      }
    };

    fetchThumbnail();
  }, [tileId, session]);

  if (isLoading) {
    return (
      <div className={`aspect-square bg-gray-100 rounded-md flex items-center justify-center ${className}`}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !imageSrc) {
    return (
      <div className={`aspect-square bg-gray-100 rounded-md flex items-center justify-center ${className}`}>
        <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
    );
  }

  return (
    <div className={`aspect-square bg-gray-100 rounded-md overflow-hidden ${className}`}>
      <Image
        src={imageSrc}
        alt={`Tile ${sku}`}
        width={200}
        height={200}
        className="w-full h-full object-cover"
      />
    </div>
  );
}


export default function TileSearch({ 
  onTileSelect,
  maxResults = 20,
  showImages = true,
  allowMultiSelect = false,
  className = ''
}: TileSearchProps) {
  const { data: session } = useSession();
  const [filters, setFilters] = useState<TileSearchFilters>({
    sku: '',
    modelName: '',
    collectionName: '',
    description: '',
    createdAfter: '',
    limit: maxResults,
    offset: 0
  });
  const [results, setResults] = useState<TileResult[]>([]);
  const [totalResults, setTotalResults] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTiles, setSelectedTiles] = useState<Set<string>>(new Set());
  const [hasSearched, setHasSearched] = useState(false);

  const updateFilter = (key: keyof TileSearchFilters, value: string | number) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setFilters({
      sku: '',
      modelName: '',
      collectionName: '',
      description: '',
      createdAfter: '',
      limit: maxResults,
      offset: 0
    });
    setResults([]);
    setTotalResults(0);
    setSelectedTiles(new Set());
    setHasSearched(false);
    setError(null);
  };

  const buildSearchPayload = () => {
    const payload: any = {
      limit: filters.limit,
      offset: filters.offset
    };

    if (filters.sku.trim()) payload.sku = filters.sku.trim();
    if (filters.modelName.trim()) payload.model_name = filters.modelName.trim();
    if (filters.collectionName.trim()) payload.collection_name = filters.collectionName.trim();
    if (filters.description.trim()) payload.description = filters.description.trim();
    if (filters.createdAfter) {
      // Convert date string to ISO datetime format for backend
      const date = new Date(filters.createdAfter);
      payload.created_after = date.toISOString();
    }

    return payload;
  };

  const performSearch = async (resetOffset: boolean = true) => {
    setIsLoading(true);
    setError(null);

    try {
      const searchFilters = resetOffset ? { ...filters, offset: 0 } : filters;
      const payload = buildSearchPayload();
      
      console.log('🔍 Starting catalog search with payload:', payload);

      const headersObj: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      
      if (session?.user?.accessToken) {
        headersObj['Authorization'] = `Bearer ${session.user.accessToken}`;
      }

      // Use Next.js API proxy route to avoid browser security restrictions
      const response = await fetch('/api/catalog-search', {
        method: 'POST',
        headers: headersObj,
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Search failed');
      }

      const data: TileSearchResponse = await response.json();
      console.log('✅ Search successful - found', data.total, 'total results,', data.results.length, 'returned');
      console.log('📊 Search results preview:', data.results.map(tile => ({
        id: tile.id,
        sku: tile.sku,
        has_image_data: tile.has_image_data
      })));
      
      if (resetOffset) {
        setResults(data.results);
        setSelectedTiles(new Set());
      } else {
        setResults(prev => [...prev, ...data.results]);
      }
      
      setTotalResults(data.total);
      setHasSearched(true);
      
      if (data.results.length === 0 && resetOffset) {
        console.info('No tiles found matching your criteria');
      }
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Search failed';
      setError(errorMessage);
      console.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMore = () => {
    if (results.length < totalResults) {
      setFilters(prev => ({ ...prev, offset: prev.offset + prev.limit }));
    }
  };

  useEffect(() => {
    if (filters.offset > 0) {
      performSearch(false);
    }
  }, [filters.offset]);

  const handleTileSelect = (tile: TileResult) => {
    if (allowMultiSelect) {
      setSelectedTiles(prev => {
        const newSet = new Set(prev);
        if (newSet.has(tile.id)) {
          newSet.delete(tile.id);
        } else {
          newSet.add(tile.id);
        }
        return newSet;
      });
    } else {
      setSelectedTiles(new Set([tile.id]));
    }
    onTileSelect?.(tile);
  };

  const hasActiveFilters = Object.entries(filters).some(([key, value]) => {
    // Exclude pagination parameters from active filters check
    if (key === 'offset' || key === 'limit') return false;
    return typeof value === 'string' ? value.trim() !== '' : value > 0;
  });

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Search Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Search Tiles</h2>
          <button
            onClick={resetFilters}
            className="text-sm text-gray-500 hover:text-gray-700"
            disabled={isLoading}
          >
            Clear all
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">SKU</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              placeholder="Search by SKU"
              value={filters.sku}
              onChange={(e) => updateFilter('sku', e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model Name</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              placeholder="Search by model name"
              value={filters.modelName}
              onChange={(e) => updateFilter('modelName', e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Collection</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              placeholder="Search by collection"
              value={filters.collectionName}
              onChange={(e) => updateFilter('collectionName', e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              placeholder="Search by description"
              value={filters.description}
              onChange={(e) => updateFilter('description', e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Created After</label>
            <input
              type="date"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              value={filters.createdAfter}
              onChange={(e) => updateFilter('createdAfter', e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Results per page</label>
            <select
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              value={filters.limit}
              onChange={(e) => updateFilter('limit', parseInt(e.target.value))}
              disabled={isLoading}
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <button
            onClick={() => performSearch()}
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            disabled={isLoading}
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>

          {hasSearched && (
            <div className="text-sm text-gray-600">
              Showing {results.length} of {totalResults} tiles
              {selectedTiles.size > 0 && ` (${selectedTiles.size} selected)`}
            </div>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <ErrorMessage 
          message={error}
          onRetry={() => performSearch()}
        />
      )}

      {/* Results */}
      {hasSearched && !error && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Search Results {totalResults > 0 && `(${totalResults} total)`}
          </h3>

          {isLoading && results.length === 0 ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner />
            </div>
          ) : results.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-lg">No tiles found</p>
              <p className="text-sm">Try adjusting your search criteria</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {results.map((tile) => (
                  <div
                    key={tile.id}
                    className={`border rounded-lg p-4 cursor-pointer transition-all ${
                      selectedTiles.has(tile.id)
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => handleTileSelect(tile)}
                  >
                    {showImages && (
                      <>
                        {tile.has_image_data ? (
                          <TileThumbnail tileId={tile.id} sku={tile.sku} className="mb-3" />
                        ) : (
                          <div className="aspect-square bg-gray-100 rounded-md flex items-center justify-center mb-3">
                            <div className="text-center text-gray-400">
                              <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                              <p className="text-xs">No Image</p>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-gray-900 truncate">{tile.sku}</h4>
                        {selectedTiles.has(tile.id) && (
                          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </div>
                      
                      <div className="text-sm text-gray-600">
                        <p><span className="font-medium">Model:</span> {tile.model_name}</p>
                        <p><span className="font-medium">Collection:</span> {tile.collection_name}</p>
                        {tile.description && (
                          <p><span className="font-medium">Description:</span> {tile.description}</p>
                        )}
                        <p><span className="font-medium">Created:</span> {new Date(tile.created_at).toLocaleDateString()}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Load More Button */}
              {results.length < totalResults && (
                <div className="flex justify-center pt-4">
                  <button
                    onClick={loadMore}
                    className="px-4 py-2 bg-gray-100 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:opacity-50"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Loading...' : `Load More (${totalResults - results.length} remaining)`}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}