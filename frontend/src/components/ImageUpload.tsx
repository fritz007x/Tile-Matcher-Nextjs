'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';

interface ImageUploadProps {
  onImageUploaded: (file: File) => void;
  isDisabled?: boolean;
}

export default function ImageUpload({ onImageUploaded, isDisabled = false }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0 || isDisabled) return;
    
    const file = acceptedFiles[0];
    setPreview(URL.createObjectURL(file));
    onImageUploaded(file);
  }, [onImageUploaded, isDisabled]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    disabled: isDisabled
  });
  
  const resetUpload = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
  };
  
  const handleRetry = () => {
    resetUpload();
  };
  
  return (
    <div className="w-full">
      {!preview ? (
        <div 
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''} ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="space-y-4">
            <div className="flex justify-center">
              <svg 
                className="w-16 h-16 text-gray-400" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24" 
                xmlns="http://www.w3.org/2000/svg"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth="2" 
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <p className="text-lg font-medium text-gray-700">
              {isDragActive
                ? 'Drop the image here...'
                : 'Drag & drop a tile image here, or click to select'}
            </p>
            <p className="text-sm text-gray-500">
              Support for JPEG, JPG, PNG, and WebP. The image should clearly show the tile pattern.
            </p>
          </div>
        </div>
      ) : (
        <div className="mt-4 border rounded-lg overflow-hidden bg-white p-4">
          <div className="aspect-video relative mb-4">
            <Image 
              src={preview} 
              alt="Uploaded tile image" 
              fill 
              className="object-contain" 
              priority
            />
          </div>
          
          <div className="flex justify-center mt-4">
            <button 
              onClick={handleRetry} 
              className="btn-secondary"
              disabled={isDisabled}
            >
              Upload Different Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
