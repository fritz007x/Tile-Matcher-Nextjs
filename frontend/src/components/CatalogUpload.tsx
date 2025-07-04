'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { toast } from 'react-hot-toast';

interface TileMetadata {
  sku: string;
  modelName: string;
  collectionName: string;
  description?: string;
}

interface UploadedFile extends File {
  preview: string;
  uploadProgress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
  metadata?: Partial<TileMetadata>;
}

interface CatalogUploadProps {
  onUploadComplete?: (results: { success: boolean; sku: string }[]) => void;
  isDisabled?: boolean;
  maxFiles?: number;
  autoUpload?: boolean;
  showMetadataForm?: boolean;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function CatalogUpload({ 
  onUploadComplete,
  isDisabled = false, 
  maxFiles = 10,
  autoUpload = true,
  showMetadataForm = true
}: CatalogUploadProps) {
  const { data: session } = useSession();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  
  // Clean up object URLs on unmount
  useEffect(() => {
    return () => {
      files.forEach(file => URL.revokeObjectURL(file.preview));
    };
  }, [files]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0 || isDisabled) return;
    
    const newFiles = acceptedFiles.slice(0, maxFiles - files.length).map(file => ({
      ...file,
      preview: URL.createObjectURL(file),
      uploadProgress: 0,
      status: 'pending' as const,
      metadata: {
        sku: '',
        modelName: '',
        collectionName: '',
        description: ''
      }
    }));
    
    setFiles(prev => [...prev, ...newFiles].slice(0, maxFiles));
  }, [onImagesUploaded, isDisabled, maxFiles]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    multiple: true,
    disabled: isDisabled || files.length >= maxFiles || isUploading
  });
  
  const updateFileMetadata = (index: number, updates: Partial<TileMetadata>) => {
    setFiles(prev => {
      const newFiles = [...prev];
      newFiles[index] = {
        ...newFiles[index],
        metadata: {
          ...newFiles[index].metadata,
          ...updates
        }
      };
      return newFiles;
    });
  };
  
  const removeFile = (index: number) => {
    setFiles(prev => {
      const newFiles = [...prev];
      URL.revokeObjectURL(newFiles[index].preview);
      newFiles.splice(index, 1);
      return newFiles;
    });
  };
  
  const resetUpload = () => {
    files.forEach(file => URL.revokeObjectURL(file.preview));
    setFiles([]);
  };
  
  const validateMetadata = (file: UploadedFile): boolean => {
    if (!file.metadata) return false;
    const { sku, modelName, collectionName } = file.metadata;
    return !!(sku && modelName && collectionName);
  };
  
  const uploadFile = async (file: UploadedFile, index: number) => {
    if (!file.metadata) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('sku', file.metadata.sku);
    formData.append('model_name', file.metadata.modelName);
    formData.append('collection_name', file.metadata.collectionName);
    if (file.metadata.description) {
      formData.append('description', file.metadata.description);
    }
    
    try {
      // Update file status to uploading
      updateFileStatus(index, 'uploading');
      
      const response = await fetch(`${API_URL}/api/matching/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session?.accessToken}`
        },
        body: formData,
      });
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'Upload failed');
      }
      
      updateFileStatus(index, 'success');
      return { success: true, sku: file.metadata.sku };
    } catch (error) {
      console.error('Upload error:', error);
      updateFileStatus(index, 'error', error instanceof Error ? error.message : 'Upload failed');
      return { success: false, sku: file.metadata.sku };
    }
  };
  
  const updateFileStatus = (index: number, status: UploadedFile['status'], error?: string) => {
    setFiles(prev => {
      const newFiles = [...prev];
      newFiles[index] = { ...newFiles[index], status, error };
      return newFiles;
    });
  };
  
  const handleUpload = async () => {
    if (!files.length) return;
    
    setIsUploading(true);
    const results = [];
    
    for (let i = 0; i < files.length; i++) {
      if (files[i].status === 'pending' && validateMetadata(files[i])) {
        const result = await uploadFile(files[i], i);
        if (result) results.push(result);
      }
    }
    
    setIsUploading(false);
    onUploadComplete?.(results);
    
    // Show success/error toast
    const successCount = results.filter(r => r.success).length;
    if (successCount > 0) {
      toast.success(`Successfully uploaded ${successCount} file(s)`);
    }
    if (results.length > successCount) {
      toast.error(`Failed to upload ${results.length - successCount} file(s)`);
    }
  };
  
  const filesRemaining = maxFiles - files.length;
  const hasPendingFiles = files.some(f => f.status === 'pending');
  
  return (
    <div className="w-full space-y-4">
      {!files.length ? (
        <div 
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''} ${
            isDisabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <input {...getInputProps()} />
          <div className="space-y-4 p-6 text-center">
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
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <p className="text-lg font-medium text-gray-700">
              {isDragActive
                ? 'Drop the catalog images here...'
                : 'Drag & drop catalog images here, or click to select'}
            </p>
            <p className="text-sm text-gray-500">
              Support for JPEG, JPG, PNG, and WebP. You can upload up to {maxFiles} images.
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {files.map((file, index) => (
              <div 
                key={index} 
                className={`relative group border rounded-lg overflow-hidden transition-all ${
                  file.status === 'error' ? 'border-red-500' : 'border-gray-200'
                }`}
              >
                {/* Image Preview */}
                <div className="aspect-square relative bg-gray-100">
                  <Image 
                    src={file.preview} 
                    alt={`Catalog image ${index + 1}`}
                    fill
                    className={`object-cover ${
                      file.status === 'uploading' ? 'opacity-50' : ''
                    }`}
                    sizes="(max-width: 640px) 100vw, (max-width: 768px) 50vw, 33vw"
                  />
                  
                  {/* Upload Progress */}
                  {file.status === 'uploading' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-200">
                      <div 
                        className="h-full bg-blue-500 transition-all duration-300"
                        style={{ width: `${file.uploadProgress}%` }}
                      />
                    </div>
                  )}
                  
                  {/* Status Overlay */}
                  <div className={`absolute inset-0 flex items-center justify-center ${
                    file.status === 'error' ? 'bg-red-50/80' : 'bg-black/20'
                  } opacity-0 group-hover:opacity-100 transition-opacity`}>
                    {file.status === 'success' && (
                      <div className="bg-green-500 text-white p-1.5 rounded-full">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    )}
                    {file.status === 'error' && (
                      <div className="text-red-600 text-sm text-center p-2">
                        <p className="font-medium">Upload Failed</p>
                        <p className="text-xs">{file.error}</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Remove Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(index);
                    }}
                    className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1.5 opacity-0 group-hover:opacity-100 transition-opacity"
                    aria-label="Remove image"
                    disabled={isUploading}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                
                {/* Metadata Form */}
                {showMetadataForm && (
                  <div className="p-3 space-y-2 bg-white">
                    <div className="space-y-1">
                      <label className="block text-xs font-medium text-gray-700">SKU *</label>
                      <input
                        type="text"
                        className="w-full text-sm border border-gray-300 rounded px-2 py-1"
                        value={file.metadata?.sku || ''}
                        onChange={(e) => updateFileMetadata(index, { sku: e.target.value })}
                        placeholder="Enter SKU"
                        disabled={file.status === 'uploading'}
                        required
                      />
                    </div>
                    
                    <div className="space-y-1">
                      <label className="block text-xs font-medium text-gray-700">Model Name *</label>
                      <input
                        type="text"
                        className="w-full text-sm border border-gray-300 rounded px-2 py-1"
                        value={file.metadata?.modelName || ''}
                        onChange={(e) => updateFileMetadata(index, { modelName: e.target.value })}
                        placeholder="Enter model name"
                        disabled={file.status === 'uploading'}
                        required
                      />
                    </div>
                    
                    <div className="space-y-1">
                      <label className="block text-xs font-medium text-gray-700">Collection *</label>
                      <input
                        type="text"
                        className="w-full text-sm border border-gray-300 rounded px-2 py-1"
                        value={file.metadata?.collectionName || ''}
                        onChange={(e) => updateFileMetadata(index, { collectionName: e.target.value })}
                        placeholder="Enter collection name"
                        disabled={file.status === 'uploading'}
                        required
                      />
                    </div>
                    
                    <div className="space-y-1">
                      <label className="block text-xs font-medium text-gray-700">Description</label>
                      <textarea
                        className="w-full text-sm border border-gray-300 rounded px-2 py-1 h-16"
                        value={file.metadata?.description || ''}
                        onChange={(e) => updateFileMetadata(index, { description: e.target.value })}
                        placeholder="Enter description (optional)"
                        disabled={file.status === 'uploading'}
                      />
                    </div>
                    
                    {file.status === 'error' && (
                      <button
                        onClick={() => uploadFile(file, index)}
                        className="w-full mt-2 text-sm text-white bg-blue-600 hover:bg-blue-700 py-1.5 px-3 rounded"
                        disabled={isUploading || !validateMetadata(file)}
                      >
                        Retry Upload
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
            
            {files.length < maxFiles && (
              <div 
                {...getRootProps()}
                className="aspect-square border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center cursor-pointer hover:border-blue-500 transition-colors"
              >
                <div className="p-4 text-center">
                  <svg 
                    className="w-8 h-8 text-gray-400 mx-auto mb-2" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M12 6v6m0 0v6m0-6h6m-6 0H6" 
                    />
                  </svg>
                  <p className="text-sm text-gray-500">
                    {isUploading ? 'Uploading...' : `Add more (${filesRemaining} remaining)`}
                  </p>
                </div>
                <input {...getInputProps()} />
              </div>
            )}
          </div>
          
          <div className="flex justify-between items-center pt-2 border-t">
            <div>
              <button 
                onClick={resetUpload}
                className="text-sm text-red-600 hover:text-red-800 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={isUploading || files.length === 0}
              >
                Clear all
              </button>
            </div>
            
            {autoUpload ? (
              <button
                onClick={handleUpload}
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={isUploading || !hasPendingFiles || files.some(f => f.status === 'pending' && !validateMetadata(f))}
              >
                {isUploading ? 'Uploading...' : `Upload ${files.filter(f => f.status === 'pending').length} Files`}
              </button>
            ) : (
              <button
                onClick={handleUpload}
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={isUploading || !hasPendingFiles || files.some(f => f.status === 'pending' && !validateMetadata(f))}
              >
                {isUploading ? 'Saving...' : 'Save All'}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
