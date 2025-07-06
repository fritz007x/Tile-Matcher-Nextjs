import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const imagePath = searchParams.get('path');

    if (!imagePath) {
      return new NextResponse('Image path is required', { status: 400 });
    }

    // Security: Prevent directory traversal
    const safePath = path.normalize(imagePath).replace(/^(\/|\\)+/, '');
    const absolutePath = path.join(process.cwd(), '..', 'backend', 'api', 'uploads', safePath);
    
    // Verify the path is within the uploads directory
    const resolvedPath = path.resolve(absolutePath);
    const uploadsDir = path.resolve(path.join(process.cwd(), '..', 'backend', 'api', 'uploads'));
    
    if (!resolvedPath.startsWith(uploadsDir)) {
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists
    try {
      await fs.access(resolvedPath);
    } catch (error) {
      return new NextResponse('Image not found', { status: 404 });
    }

    // Read the image file
    const imageBuffer = await fs.readFile(resolvedPath);
    const ext = path.extname(resolvedPath).toLowerCase().substring(1);
    
    // Determine content type based on file extension
    const contentType = {
      'jpg': 'image/jpeg',
      'jpeg': 'image/jpeg',
      'png': 'image/png',
      'gif': 'image/gif',
      'webp': 'image/webp',
      'bmp': 'image/bmp',
      'tiff': 'image/tiff'
    }[ext] || 'application/octet-stream';

    // Return the image with appropriate headers
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=31536000, immutable'
      }
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
