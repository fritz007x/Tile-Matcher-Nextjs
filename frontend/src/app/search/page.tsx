'use client';

import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';
import TileSearch from '@/components/TileSearch';

export default function SearchPage() {
  const { data: session, status } = useSession();

  if (status === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!session) {
    redirect('/login');
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Search Catalog</h1>
          <p className="mt-2 text-gray-600">
            Search and filter through your tile catalog using various criteria like SKU, model name, collection, and more.
          </p>
        </div>

        {/* Search Component */}
        <TileSearch
          showImages={true}
          maxResults={20}
          onTileSelect={(tile) => console.log('Selected tile:', tile)}
          className=""
        />
      </div>
    </div>
  );
}