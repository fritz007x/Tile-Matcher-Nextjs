'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import LoadingSpinner from '@/components/LoadingSpinner';
import { apiService } from '@/lib/api';
import { MatchResultItem } from '@/types';
import AuthService from '@/lib/auth';

interface MatchDetailPageProps {
  params: {
    id: string;
  }
}

export default function MatchDetailPage({ params }: MatchDetailPageProps) {
  const router = useRouter();
  const { id } = params;
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [isSaved, setIsSaved] = useState(false);
  const [match, setMatch] = useState<{
    id: string;
    date: string;
    queryImageUrl: string;
    results: MatchResultItem[];
  } | null>(null);

  useEffect(() => {
    // Check if user is authenticated
    if (!AuthService.isAuthenticated()) {
      router.push('/login');
      return;
    }

    const fetchMatchDetails = async () => {
      setIsLoading(true);
      try {
        // In a real app, fetch match details from API
        // const data = await apiService.matching.getMatchById(id);
        
        // Mock data for demonstration
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Mock match details
        setMatch({
          id,
          date: '2025-06-20T15:30:00Z',
          queryImageUrl: '/placeholder-tile.jpg',
          results: [
            {
              tile_id: 'tile123',
              score: 0.95,
              method: 'ViT',
              metadata: {
                sku: 'TL-1234',
                model_name: 'Carrara Marble',
                collection_name: 'Italian Collection',
                image_url: '/placeholder-tile.jpg'
              }
            },
            {
              tile_id: 'tile456',
              score: 0.87,
              method: 'SIFT',
              metadata: {
                sku: 'TL-5678',
                model_name: 'Terracotta Classic',
                collection_name: 'Mediterranean Series',
                image_url: '/placeholder-tile-2.jpg'
              }
            },
            {
              tile_id: 'tile789',
              score: 0.82,
              method: 'ORB',
              metadata: {
                sku: 'TL-9012',
                model_name: 'Slate Gray',
                collection_name: 'Modern Collection',
                image_url: '/placeholder-tile-3.jpg'
              }
            }
          ]
        });
        
        // Check if this match is saved (would be an API call in real app)
        setIsSaved(Math.random() > 0.5);
        
      } catch (err: any) {
        console.error('Error fetching match details:', err);
        setError('Failed to load match details. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchMatchDetails();
  }, [id, router]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleSaveToggle = async () => {
    try {
      // In a real app, call API to save/unsave match
      // await apiService.user.toggleSaveMatch(id);
      
      // For demo, just toggle the state
      setIsSaved(prev => !prev);
    } catch (err) {
      console.error('Error toggling save status:', err);
    }
  };

  const handleRequestSample = async (sku: string) => {
    try {
      // In a real app, call API to request sample
      // await apiService.catalog.requestSample(sku);
      
      // For demo, just show an alert
      alert(`Sample requested for SKU: ${sku}`);
    } catch (err) {
      console.error('Error requesting sample:', err);
    }
  };

  if (isLoading) {
    return (
      <main className="flex min-h-screen flex-col">
        <Header />
        <div className="flex-grow flex items-center justify-center">
          <LoadingSpinner size="large" text="Loading match details..." />
        </div>
        <Footer />
      </main>
    );
  }

  if (error || !match) {
    return (
      <main className="flex min-h-screen flex-col">
        <Header />
        <div className="container mx-auto px-4 py-16 flex-grow">
          <div className="max-w-lg mx-auto text-center">
            <h1 className="text-3xl font-bold text-red-600 mb-4">Error</h1>
            <div className="bg-white shadow-md rounded-lg p-8 mb-8">
              <p className="text-lg text-gray-700 mb-6">
                {error || "Match not found or has been removed."}
              </p>
              <Link href="/" className="btn-primary">
                Return to Homepage
              </Link>
            </div>
          </div>
        </div>
        <Footer />
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col">
      <Header />
      
      <div className="container mx-auto px-4 py-8 flex-grow">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-1">Match Results</h1>
            <p className="text-gray-600">
              Searched on {formatDate(match.date)}
            </p>
          </div>
          <div className="flex space-x-4">
            <button
              onClick={handleSaveToggle}
              className={`flex items-center px-4 py-2 rounded-md transition ${
                isSaved 
                  ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <svg className="w-5 h-5 mr-2" fill={isSaved ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={isSaved ? 0 : 2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
              {isSaved ? 'Saved' : 'Save'}
            </button>
            <Link href="/history" className="btn-outline">
              Back to History
            </Link>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Uploaded image */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Your Uploaded Tile</h2>
              <div className="aspect-square relative rounded-md overflow-hidden mb-6">
                <Image 
                  src={match.queryImageUrl}
                  alt="Uploaded tile" 
                  fill 
                  className="object-cover"
                  placeholder="blur"
                  blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Match ID:</span>
                  <span className="font-medium">{match.id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Date:</span>
                  <span>{formatDate(match.date)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Matches:</span>
                  <span>{match.results.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Best Match Score:</span>
                  <span className="font-medium text-green-700">
                    {(match.results[0].score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Match results */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow">
              <div className="border-b px-6 py-4">
                <h2 className="text-xl font-semibold">Best Matching Tiles</h2>
              </div>
              
              <div className="divide-y">
                {match.results.map((result, index) => (
                  <div key={result.tile_id} className="p-6">
                    <div className="flex flex-col md:flex-row gap-6">
                      <div className="w-full md:w-1/3">
                        <div className="aspect-square relative rounded-md overflow-hidden">
                          <Image 
                            src={result.metadata.image_url}
                            alt={result.metadata.model_name} 
                            fill 
                            className="object-cover"
                            placeholder="blur"
                            blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                          />
                        </div>
                      </div>
                      <div className="w-full md:w-2/3 flex flex-col">
                        <div className="flex items-start justify-between">
                          <h3 className="text-xl font-semibold">{result.metadata.model_name}</h3>
                          <span 
                            className={`inline-block text-sm font-medium px-3 py-1 rounded-full ${
                              result.score > 0.9 
                                ? 'bg-green-100 text-green-800' 
                                : result.score > 0.8 
                                ? 'bg-blue-100 text-blue-800'
                                : 'bg-yellow-100 text-yellow-800'
                            }`}
                          >
                            {(result.score * 100).toFixed(1)}% Match
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-4">
                          <div>
                            <span className="block text-sm text-gray-500">SKU</span>
                            <span className="font-medium">{result.metadata.sku}</span>
                          </div>
                          <div>
                            <span className="block text-sm text-gray-500">Collection</span>
                            <span className="font-medium">{result.metadata.collection_name}</span>
                          </div>
                          <div>
                            <span className="block text-sm text-gray-500">Match Method</span>
                            <span className="font-medium">{result.method}</span>
                          </div>
                          <div>
                            <span className="block text-sm text-gray-500">Price Range</span>
                            <span className="font-medium">$24.99 - $29.99/sq.ft</span>
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-3 mt-auto pt-4">
                          <button 
                            onClick={() => handleRequestSample(result.metadata.sku)}
                            className="btn-primary"
                          >
                            Request Sample
                          </button>
                          <a 
                            href={`/catalog/${result.metadata.sku}`} 
                            className="btn-outline"
                            target="_blank" 
                            rel="noopener noreferrer"
                          >
                            View in Catalog
                          </a>
                        </div>
                      </div>
                    </div>
                    
                    {index === 0 && (
                      <div className="mt-6 p-4 bg-blue-50 rounded-md">
                        <h4 className="font-medium text-blue-800 mb-2">Perfect Match!</h4>
                        <p className="text-blue-700 text-sm">
                          This tile is an excellent match to your uploaded image with a {(result.score * 100).toFixed(1)}% similarity score. 
                          The match was determined using our advanced {result.method} algorithm.
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            <div className="mt-6 bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Technical Information</h2>
              <div className="space-y-4">
                <p className="text-gray-700">
                  This match was performed using multiple algorithms for accuracy. The primary match was made using the {match.results[0].method} algorithm, 
                  which found a {(match.results[0].score * 100).toFixed(1)}% similarity to your uploaded image.
                </p>
                <div className="mt-4">
                  <h3 className="font-medium mb-2">Matching Algorithms Used:</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-3 bg-gray-100 rounded-md">
                      <span className="block font-medium">SIFT</span>
                      <span className="text-sm text-gray-600">Feature detection</span>
                    </div>
                    <div className="p-3 bg-gray-100 rounded-md">
                      <span className="block font-medium">ORB</span>
                      <span className="text-sm text-gray-600">Fast matching</span>
                    </div>
                    <div className="p-3 bg-gray-100 rounded-md">
                      <span className="block font-medium">ViT</span>
                      <span className="text-sm text-gray-600">Deep learning</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
