'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function ErrorPage({ error, reset }: ErrorPageProps) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Application error:', error);
  }, [error]);

  return (
    <main className="flex min-h-screen flex-col">
      <Header />
      
      <div className="container mx-auto px-4 py-16 flex-grow">
        <div className="max-w-lg mx-auto text-center">
          <h1 className="text-4xl font-bold text-red-600 mb-4">
            Something went wrong
          </h1>
          
          <div className="bg-white shadow-md rounded-lg p-8 mb-8">
            <p className="text-lg text-gray-700 mb-6">
              We're sorry, but we encountered an error while processing your request.
            </p>
            
            {process.env.NODE_ENV === 'development' && (
              <div className="bg-gray-100 p-4 rounded-md mb-6 text-left">
                <p className="font-mono text-sm text-red-800 break-all">
                  {error.message || 'Unknown error occurred'}
                </p>
                {error.digest && (
                  <p className="font-mono text-xs text-gray-600 mt-2">
                    Error ID: {error.digest}
                  </p>
                )}
              </div>
            )}
            
            <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 justify-center">
              <button
                onClick={reset}
                className="btn-primary"
              >
                Try Again
              </button>
              
              <Link href="/" className="btn-secondary">
                Go to Homepage
              </Link>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
