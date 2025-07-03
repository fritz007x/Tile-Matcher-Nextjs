import { User } from '@/types';
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { signIn, signOut, useSession } from 'next-auth/react';

// AuthService for handling token storage and user authentication
export const AuthService = {
  // Google sign-in method
  signInWithGoogle: async (): Promise<boolean> => {
    try {
      const result = await signIn('google', { redirect: false });
      return !result?.error;
    } catch (error) {
      console.error('Google sign-in error:', error);
      return false;
    }
  },
  
  // Sign out method (works for all auth methods)
  signOut: async (): Promise<void> => {
    await signOut({ redirect: false });
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
  },
  // Get the stored token
  getToken: (): string | null => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('token');
    }
    return null;
  },

  // Set the authentication token
  setToken: (token: string): void => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
    }
  },

  // Remove the stored token
  removeToken: (): void => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
    }
  },

  // Check if user is authenticated
  isAuthenticated: (): boolean => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('token') !== null;
    }
    return false;
  },

  // Save user data
  setUser: (user: User): void => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('user', JSON.stringify(user));
    }
  },

  // Get user data
  getUser: (): User | null => {
    if (typeof window !== 'undefined') {
      const userStr = localStorage.getItem('user');
      if (userStr) {
        try {
          return JSON.parse(userStr);
        } catch (e) {
          console.error('Error parsing user data', e);
          return null;
        }
      }
    }
    return null;
  },

  // Clear all auth data (logout)
  clearAuth: (): void => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
  }
};

// Hook to protect routes that require authentication
export const withAuth = <T extends object>(Component: React.ComponentType<T>) => {
  return function ProtectedRoute(props: T) {
    const [isClient, setIsClient] = useState(false);
    const router = useRouter();
    const { data: session, status } = useSession();
    
    // Handle client-side rendering
    useEffect(() => {
      setIsClient(true);
      
      // Check if user is authenticated (either via NextAuth session or local token)
      if (status === 'unauthenticated' && !AuthService.isAuthenticated()) {
        // Redirect to login page with return URL
        const returnPath = window.location.pathname;
        router.push(`/login?returnUrl=${encodeURIComponent(returnPath)}`);
      }
    }, [router, status]);
    
    // If not client yet or still loading session, show loading state
    if (!isClient || status === 'loading') {
      return React.createElement('div', { className: "flex justify-center p-8" },
        React.createElement('span', { className: "loading loading-spinner" })
      );
    }
    
    // If client and got past the auth check in useEffect, render the component
    return (status === 'authenticated' || AuthService.isAuthenticated()) 
      ? React.createElement(Component, props) 
      : null;
  };
};

export default AuthService;
