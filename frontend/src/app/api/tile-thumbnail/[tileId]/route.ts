import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { tileId: string } }
) {
  try {
    const { tileId } = params;
    const { searchParams } = new URL(request.url);
    const width = searchParams.get('width') || '200';
    const height = searchParams.get('height') || '200';
    
    console.log(`🖼️ Thumbnail request for tile ${tileId} (${width}x${height})`);
    
    // Get authorization header from the original request
    const authHeader = request.headers.get('authorization');
    const headers: Record<string, string> = {};
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
    
    const response = await fetch(
      `http://localhost:8000/api/matching/tile/${tileId}/thumbnail?width=${width}&height=${height}`,
      {
        headers,
        signal: AbortSignal.timeout(10000) // 10 second timeout
      }
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Thumbnail fetch failed for ${tileId}: ${response.status}`, errorText);
      return NextResponse.json({
        detail: `Backend thumbnail fetch failed: ${response.status}`,
        error: errorText
      }, { status: response.status });
    }
    
    const data = await response.json();
    console.log(`✅ Thumbnail fetched successfully for ${tileId}`, {
      content_type: data.content_type,
      has_data: !!data.data,
      data_length: data.data?.length
    });
    
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('❌ Thumbnail proxy error:', error);
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json({
        detail: 'Thumbnail request timeout',
        error: 'timeout'
      }, { status: 408 });
    }
    
    return NextResponse.json({
      detail: 'Failed to fetch thumbnail',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}