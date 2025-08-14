import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    console.log('🔍 Catalog search proxy - processing request...');
    
    const searchPayload = await request.json();
    console.log('📋 Search parameters:', searchPayload);
    
    // Get authorization header from the original request
    const authHeader = request.headers.get('authorization');
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
    
    const startTime = Date.now();
    const response = await fetch('http://localhost:8000/api/matching/search', {
      method: 'POST',
      headers,
      body: JSON.stringify(searchPayload),
      signal: AbortSignal.timeout(30000) // 30 second timeout for search
    });
    const endTime = Date.now();
    
    console.log('⏱️ Backend search response in:', (endTime - startTime) / 1000, 'seconds');
    console.log('📊 Backend status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Backend search error:', errorText);
      
      try {
        const errorJson = JSON.parse(errorText);
        return NextResponse.json(errorJson, { status: response.status });
      } catch {
        return NextResponse.json(
          { detail: `Backend search failed: ${response.status}` }, 
          { status: response.status }
        );
      }
    }
    
    const result = await response.json();
    console.log('✅ Backend search success - found', result.total || 0, 'total results');
    
    // Return the exact same format as the backend
    return NextResponse.json(result);
    
  } catch (error) {
    console.error('❌ API route search failed:', error);
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json({
        detail: 'Search timeout - the search took too long. Please try again.',
        error: 'timeout'
      }, { status: 408 });
    }
    
    return NextResponse.json({
      detail: 'Search failed',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}