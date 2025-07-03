import config from '@/config';

/**
 * Utility functions for image handling in the frontend
 */

/**
 * Validates if a file is an acceptable image format and size
 * @param file The file to validate
 * @returns An object with validation result and error message if any
 */
export const validateImage = (file: File): { isValid: boolean; error?: string } => {
  // Check file size
  const maxSizeInBytes = config.imageUploadSizeLimit * 1024 * 1024; // Convert MB to bytes
  if (file.size > maxSizeInBytes) {
    return {
      isValid: false,
      error: `File size exceeds the ${config.imageUploadSizeLimit}MB limit.`
    };
  }
  
  // Check file type
  const fileExtension = file.name.split('.').pop()?.toLowerCase();
  if (!fileExtension || !config.supportedImageFormats.includes(fileExtension)) {
    return {
      isValid: false,
      error: `Unsupported file format. Supported formats: ${config.supportedImageFormats.join(', ')}`
    };
  }
  
  return { isValid: true };
};

/**
 * Compresses an image file to reduce its size before upload
 * @param file The image file to compress
 * @param maxWidthOrHeight Maximum width or height in pixels
 * @param quality Compression quality (0-1)
 * @returns Promise resolving to a compressed Blob
 */
export const compressImage = async (
  file: File, 
  maxWidthOrHeight = 1200, 
  quality = 0.8
): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target?.result as string;
      
      img.onload = () => {
        // Calculate new dimensions while maintaining aspect ratio
        let width = img.width;
        let height = img.height;
        
        if (width > height) {
          if (width > maxWidthOrHeight) {
            height = Math.round(height * maxWidthOrHeight / width);
            width = maxWidthOrHeight;
          }
        } else {
          if (height > maxWidthOrHeight) {
            width = Math.round(width * maxWidthOrHeight / height);
            height = maxWidthOrHeight;
          }
        }
        
        // Create canvas and compress image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }
        
        ctx.drawImage(img, 0, 0, width, height);
        
        // Get the compressed image as blob
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error('Could not compress image'));
              return;
            }
            resolve(blob);
          },
          file.type,
          quality
        );
      };
      
      img.onerror = () => {
        reject(new Error('Error loading image'));
      };
    };
    
    reader.onerror = () => {
      reject(new Error('Error reading file'));
    };
  });
};

/**
 * Creates an object URL for a file for preview
 * @param file The file to create a URL for
 * @returns The object URL
 */
export const createObjectURL = (file: File): string => {
  return URL.createObjectURL(file);
};

/**
 * Revokes an object URL to free up memory
 * @param url The URL to revoke
 */
export const revokeObjectURL = (url: string): void => {
  URL.revokeObjectURL(url);
};

/**
 * Extracts the EXIF data from an image if available
 * @param file The image file
 * @returns Promise resolving to EXIF data or null if not available
 */
export const extractExifData = async (file: File): Promise<any | null> => {
  // This is a placeholder function
  // In a real implementation, you would use a library like exif-js to extract metadata
  return null;
};
