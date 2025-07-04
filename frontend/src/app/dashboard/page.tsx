'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import LoadingSpinner from '@/components/LoadingSpinner';
import { MatchResultItem } from '@/types/match';
import { withAuth } from '@/lib/auth';

interface SavedMatch {
  id: string;
  date: string;
  imageUrl: string;
  results: MatchResultItem[];
}

function DashboardPage() {
  const { data: session } = useSession();
  const [isLoading, setIsLoading] = useState(true);
  const [recentMatches, setRecentMatches] = useState<SavedMatch[]>([]);
  const [savedMatches, setSavedMatches] = useState<SavedMatch[]>([]);

  // Mock data for demonstration
  const mockMatches: SavedMatch[] = [
    {
      id: '1',
      date: '2025-06-20',
      imageUrl: '/placeholder-tile.jpg',
      results: [
        {
          tile_id: 'tile123',
          imageUrl: '/placeholder-tile.jpg',
          similarity: 95,
          score: 0.95,
          method: 'ViT',
          metadata: {
            sku: 'TL-1234',
            model_name: 'Carrara Marble',
            collection_name: 'Italian Collection',
            image_url: '/placeholder-tile.jpg'
          }
        }
      ]
    },
    {
      id: '2',
      date: '2025-06-18',
      imageUrl: '/placeholder-tile-2.jpg',
      results: [
        {
          tile_id: 'tile456',
          imageUrl: '/placeholder-tile-2.jpg',
          similarity: 87,
          score: 0.87,
          method: 'SIFT',
          metadata: {
            sku: 'TL-5678',
            model_name: 'Terracotta Classic',
            collection_name: 'Mediterranean Series',
            image_url: '/placeholder-tile-2.jpg'
          }
        }
      ]
    }
  ];

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Using mock data for now
        setRecentMatches(mockMatches);
        setSavedMatches(mockMatches.slice(0, 1));
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <main className="flex min-h-screen flex-col">
      <Header />
      
      <div className="container mx-auto px-4 py-8 flex-grow">
        <h1 className="text-3xl font-bold mb-6">Your Dashboard</h1>
        
        {isLoading ? (
          <div className="py-12">
            <LoadingSpinner text="Loading your dashboard..." />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left sidebar with user info */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow p-6 mb-6">
                <h2 className="text-xl font-semibold mb-4">Account Information</h2>
                <div className="space-y-3">
                  <p className="text-gray-700">
                    <span className="font-medium block">Name:</span>
                    {session?.user?.name || 'User'}
                  </p>
                  <p className="text-gray-700">
                    <span className="font-medium block">Email:</span>
                    {session?.user?.email || 'user@example.com'}
                  </p>
                  <p className="text-gray-700">
                    <span className="font-medium block">Account Type:</span>
                    Free
                  </p>
                </div>
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <Link href="/settings" className="btn-secondary w-full">
                    Account Settings
                  </Link>
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Usage Statistics</h2>
                <div className="space-y-3">
                  <div>
                    <span className="font-medium">Total Searches:</span>
                    <span className="float-right">24</span>
                  </div>
                  <div>
                    <span className="font-medium">Saved Matches:</span>
                    <span className="float-right">{savedMatches.length}</span>
                  </div>
                  <div>
                    <span className="font-medium">Search Accuracy:</span>
                    <span className="float-right">92%</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Main content area */}
            <div className="lg:col-span-2 space-y-6">
              {/* Recent searches */}
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold">Recent Matches</h2>
                  <Link href="/history" className="text-blue-600 text-sm hover:text-blue-800">
                    View All
                  </Link>
                </div>
                
                {recentMatches.length === 0 ? (
                  <p className="text-gray-500 py-4">You haven't performed any matches yet.</p>
                ) : (
                  <div className="space-y-4">
                    {recentMatches.map(match => (
                      <div key={match.id} className="border rounded-lg p-4 flex items-center">
                        <div className="w-20 h-20 relative mr-4 flex-shrink-0">
                          <Image 
                            src={match.imageUrl}
                            alt="Matched tile" 
                            fill 
                            className="object-cover rounded"
                            placeholder="blur"
                            blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                          />
                        </div>
                        <div className="flex-grow">
                          <p className="text-sm text-gray-500">
                            Matched on {formatDate(match.date)}
                          </p>
                          <p className="font-medium">
                            Top match: {match.results[0].metadata.model_name}
                          </p>
                          <p className="text-sm text-gray-600">
                            Collection: {match.results[0].metadata.collection_name}
                          </p>
                          <div className="mt-2">
                            <Link href={`/matches/${match.id}`} className="text-blue-600 text-sm hover:text-blue-800">
                              View Details
                            </Link>
                          </div>
                        </div>
                        <div className="flex-shrink-0 text-right">
                          <span className="inline-block bg-green-100 text-green-800 text-sm font-medium px-2.5 py-1 rounded">
                            {(match.results[0].score * 100).toFixed(1)}% Match
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Saved matches */}
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold">Saved Matches</h2>
                  <Link href="/saved" className="text-blue-600 text-sm hover:text-blue-800">
                    View All
                  </Link>
                </div>
                
                {savedMatches.length === 0 ? (
                  <p className="text-gray-500 py-4">No saved matches found.</p>
                ) : (
                  <div className="space-y-4">
                    {savedMatches.map(match => (
                      <div key={match.id} className="border rounded-lg p-4 flex items-center">
                        <div className="w-20 h-20 relative mr-4 flex-shrink-0">
                          <Image 
                            src={match.imageUrl}
                            alt="Saved tile" 
                            fill 
                            className="object-cover rounded"
                            placeholder="blur"
                            blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                          />
                        </div>
                        <div className="flex-grow">
                          <p className="text-sm text-gray-500">
                            Saved on {formatDate(match.date)}
                          </p>
                          <p className="font-medium">
                            {match.results[0].metadata.model_name}
                          </p>
                          <p className="text-sm text-gray-600">
                            SKU: {match.results[0].metadata.sku}
                          </p>
                          <div className="mt-2">
                            <Link href={`/matches/${match.id}`} className="text-blue-600 text-sm hover:text-blue-800">
                              View Details
                            </Link>
                          </div>
                        </div>
                        <div className="flex-shrink-0">
                          <button className="text-red-500 hover:text-red-700">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
      
      <Footer />
    </main>
  );
}

export default withAuth(DashboardPage);
