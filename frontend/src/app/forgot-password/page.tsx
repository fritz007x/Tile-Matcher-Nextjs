'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { apiService } from '@/lib/api';
import LoadingSpinner from '@/components/LoadingSpinner';
import { ApiError } from '@/types';

export default function ForgotPasswordPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    if (!email) {
      setError('Please enter your email address');
      return;
    }
    
    // Simple email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address');
      return;
    }
    
    try {
      setIsLoading(true);
      setError('');
      
      // Call the password reset API
      try {
        await apiService.auth.forgotPassword({ email });
        setIsSubmitted(true);
      } catch (err: unknown) {
        const apiError = err as ApiError;
        if (apiError.status === 404) {
          // Don't reveal if email exists or not (security best practice)
          // Still show success message even if email not found
          setIsSubmitted(true);
        } else {
          setError(apiError.message || 'An error occurred. Please try again.');
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process your request. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col">
      <Header />
      
      <div className="flex-grow container mx-auto px-4 py-12">
        <div className="max-w-md mx-auto">
          <h1 className="text-3xl font-bold text-center mb-8">Reset Your Password</h1>
          
          <div className="bg-white rounded-xl shadow-md overflow-hidden p-6">
            {isSubmitted ? (
              <div className="text-center py-6">
                <div className="mb-4 text-green-600">
                  <svg className="h-16 w-16 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold mb-2">Check Your Email</h3>
                <p className="text-gray-600 mb-6">
                  <span className="font-medium">Reset link sent!</span> Please check your email inbox for instructions to reset your password. If you don't receive an email within a few minutes, please check your spam folder.
                </p>
                <div className="flex justify-center">
                  <Link href="/login" className="btn-outline">
                    Return to Login
                  </Link>
                </div>
              </div>
            ) : (
              <>
                <p className="text-gray-600 mb-6">
                  Enter your email address and we'll send you instructions to reset your password.
                </p>
                
                <form onSubmit={handleSubmit} className="space-y-5">
                  {error && (
                    <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">
                      {error}
                    </div>
                  )}
                  
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                      Email Address
                    </label>
                    <input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="your@email.com"
                      required
                    />
                  </div>
                  
                  <div className="pt-2">
                    <button
                      type="submit"
                      className="w-full btn-primary py-2.5"
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <LoadingSpinner size="small" text="" />
                          <span className="ml-2">Sending...</span>
                        </>
                      ) : (
                        'Send Reset Instructions'
                      )}
                    </button>
                  </div>
                </form>
                
                <div className="mt-6 text-center">
                  <p className="text-gray-600">
                    Remember your password?{' '}
                    <Link href="/login" className="text-blue-600 hover:text-blue-800 font-medium">
                      Sign in
                    </Link>
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
      
      <Footer />
    </main>
  );
}
