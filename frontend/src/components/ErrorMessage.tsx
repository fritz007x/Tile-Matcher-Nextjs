'use client';

import { ApiError } from '@/types';

interface ErrorMessageProps {
  error: string | ApiError | null;
  onRetry?: () => void;
  variant?: 'inline' | 'card';
  retryText?: string;
  className?: string;
  showDetails?: boolean;
}

export default function ErrorMessage({
  error,
  onRetry,
  variant = 'inline',
  retryText = 'Retry',
  className = '',
  showDetails = false
}: ErrorMessageProps) {
  if (!error) return null;

  // Extract error message and details
  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorCode = typeof error === 'object' ? error.errorCode : undefined;
  const errorDetails = typeof error === 'object' ? error.details : undefined;

  const content = (
    <div className={`flex flex-col ${variant === 'card' ? 'p-4' : ''}`}>
      <div className="flex items-start">
        <svg
          className="h-5 w-5 text-red-500 mr-2 mt-0.5 flex-shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <div className="flex-1">
          <p className="text-sm text-red-700">{errorMessage}</p>
          {errorCode && showDetails && (
            <p className="text-xs text-red-600 mt-1">Error Code: {errorCode}</p>
          )}
          {errorDetails && showDetails && Object.keys(errorDetails).length > 0 && (
            <div className="mt-2 text-xs text-red-600">
              {Object.entries(errorDetails).map(([key, value]) => (
                <div key={key}>
                  <strong>{key}:</strong> {String(value)}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 self-start"
        >
          {retryText}
        </button>
      )}
    </div>
  );

  if (variant === 'card') {
    return (
      <div className={`bg-red-50 border-l-4 border-red-400 p-4 ${className}`}>
        {content}
      </div>
    );
  }

  return <div className={className}>{content}</div>;
}
