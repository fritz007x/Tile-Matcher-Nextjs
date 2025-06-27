import Link from 'next/link';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col">
      <Header />
      
      <div className="container mx-auto px-4 py-16 flex-grow">
        <div className="max-w-lg mx-auto text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            404 - Page Not Found
          </h1>
          
          <div className="bg-white shadow-md rounded-lg p-8 mb-8">
            <p className="text-lg text-gray-700 mb-6">
              The page you are looking for does not exist or has been moved.
            </p>
            
            <svg 
              className="w-32 h-32 mx-auto text-gray-400 mb-6" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth="1.5" 
                d="M12 14l9-5-9-5-9 5 9 5z" 
              />
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth="1.5" 
                d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" 
              />
              <path 
                strokeLinecap="round"
                strokeLinejoin="round" 
                strokeWidth="1.5"
                d="M12 14l9-5-9-5-9 5 9 5zm0 0l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14zm-4 6v-7.5l4-2.222" 
              />
            </svg>
            
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
