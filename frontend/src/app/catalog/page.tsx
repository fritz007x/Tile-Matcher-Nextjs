'use client';

import { useState } from 'react';
import CatalogUpload from '@/components/CatalogUpload';
import TileSearch from '@/components/TileSearch';
import { useSession } from 'next-auth/react';
import { redirect } from 'next/navigation';

export default function CatalogPage() {
  const { data: session, status } = useSession();
  const [uploadResults, setUploadResults] = useState<{ success: boolean; sku: string }[]>([]);
  const [activeTab, setActiveTab] = useState<'upload' | 'search'>('upload');

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

  const handleUploadComplete = (results: { success: boolean; sku: string }[]) => {
    setUploadResults(results);
    console.log('Upload results:', results);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Catalog Management</h1>
          <p className="mt-2 text-gray-600">
            Upload and manage your tile catalog images. Search and filter through your catalog using various criteria.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('upload')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'upload'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Upload Tiles
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'search'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Search Catalog
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'upload' && (
          <>
            {/* Upload Section */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Catalog Images</h2>
              <CatalogUpload
                onUploadComplete={handleUploadComplete}
                maxFiles={20}
                autoUpload={true}
                showMetadataForm={true}
              />
            </div>

            {/* Upload Results Section */}
            {uploadResults.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Results</h2>
                <div className="space-y-2">
                  {uploadResults.map((result, index) => (
                    <div
                      key={index}
                      className={`flex items-center justify-between p-3 rounded-md ${
                        result.success
                          ? 'bg-green-50 border border-green-200'
                          : 'bg-red-50 border border-red-200'
                      }`}
                    >
                      <div className="flex items-center">
                        {result.success ? (
                          <svg className="w-5 h-5 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        ) : (
                          <svg className="w-5 h-5 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        )}
                        <span className="font-medium">SKU: {result.sku}</span>
                      </div>
                      <span
                        className={`text-sm font-medium ${
                          result.success ? 'text-green-700' : 'text-red-700'
                        }`}
                      >
                        {result.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Instructions */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <h3 className="text-lg font-medium text-blue-900 mb-2">Upload Guidelines</h3>
              <ul className="text-blue-800 space-y-1 text-sm">
                <li>• Each image should clearly show the tile pattern and texture</li>
                <li>• Use high-quality images (minimum 512x512 pixels recommended)</li>
                <li>• Ensure good lighting and minimal shadows</li>
                <li>• Fill in all required metadata fields (SKU, Model Name, Collection)</li>
                <li>• SKUs should be unique - existing tiles will be updated if SKU matches</li>
              </ul>
            </div>
          </>
        )}

        {activeTab === 'search' && (
          <TileSearch
            showImages={true}
            maxResults={20}
            onTileSelect={(tile) => console.log('Selected tile:', tile)}
            className=""
          />
        )}
      </div>
    </div>
  );
}
