'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { useSession, signOut } from 'next-auth/react';

export default function Header() {
  const { data: session, status } = useSession();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleSignOut = async () => {
    await signOut({ callbackUrl: '/' });
  };
  
  return (
    <header className="bg-white shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-4">
          {/* Logo and site name */}
          <div className="flex items-center space-x-2">
            <Link href="/" className="flex items-center">
              <div className="font-bold text-xl text-blue-600">Tile Matcher</div>
            </Link>
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <Link href="/" className="text-gray-700 hover:text-blue-600 transition-colors">
              Home
            </Link>
            <Link href="/about" className="text-gray-700 hover:text-blue-600 transition-colors">
              About
            </Link>
            {status === 'authenticated' ? (
              <>
                <Link href="/dashboard" className="text-gray-700 hover:text-blue-600 transition-colors">
                  Dashboard
                </Link>
                <button onClick={handleSignOut} className="btn-secondary">
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Link href="/login" className="text-gray-700 hover:text-blue-600 transition-colors">
                  Login
                </Link>
                <Link href="/register" className="btn-primary">
                  Sign Up
                </Link>
              </>
            )}
          </nav>
          
          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className="text-gray-500 hover:text-gray-700 focus:outline-none"
              onClick={toggleMenu}
            >
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMenuOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden bg-white border-t">
          <div className="container mx-auto px-4 py-4 space-y-4">
            <Link href="/" className="block py-2 text-gray-700 hover:text-blue-600 transition-colors" onClick={toggleMenu}>Home</Link>
            <Link href="/about" className="block py-2 text-gray-700 hover:text-blue-600 transition-colors" onClick={toggleMenu}>About</Link>
            {status === 'authenticated' ? (
              <>
                <Link href="/dashboard" className="block py-2 text-gray-700 hover:text-blue-600 transition-colors" onClick={toggleMenu}>Dashboard</Link>
                <button 
                  onClick={() => { toggleMenu(); handleSignOut(); }}
                  className="w-full text-left py-2 text-red-600 font-medium"
                >
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Link href="/login" className="block py-2 text-gray-700 hover:text-blue-600 transition-colors" onClick={toggleMenu}>Login</Link>
                <Link href="/register" className="block w-full text-center btn-primary py-2" onClick={toggleMenu}>Sign Up</Link>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
