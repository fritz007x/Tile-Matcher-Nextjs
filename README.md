# Tile Matcher Application

A powerful image matching application that helps users identify tiles by comparing photos against a catalog of tile images using advanced computer vision techniques. The application combines both classical computer vision algorithms and deep learning approaches for accurate and robust tile matching.

## Features

- **Advanced Image Matching**: Multiple matching techniques (SIFT, ORB, KAZE, and Vision Transformers)
- **High Accuracy**: Ensemble approach combining results from multiple methods
- **Fast Search**: FAISS vector indexing for efficient similarity search
- **User Authentication**: Secure login with token-based authentication
- **User Dashboard**: Track matching history and save favorite matches
- **Responsive UI**: Optimized for desktop with mobile support
- **Image Processing**: Client-side image compression and validation

## Tech Stack

### Frontend
- **Framework**: Next.js 14 with TypeScript and App Router
- **Styling**: Tailwind CSS with custom components
- **State Management**: React hooks and context
- **API Integration**: Axios for API requests
- **Image Handling**: Client-side image processing and preview
- **Form Handling**: Native form handling with validation

### Backend
- **API Framework**: FastAPI (Python)
- **Image Processing**: OpenCV (SIFT, ORB, KAZE algorithms)
- **Deep Learning**: Vision Transformer (ViT) via HuggingFace Transformers
- **Vector Search**: FAISS for efficient similarity search
- **Database**: MongoDB for metadata storage
- **Authentication**: JWT token-based auth with OAuth2PasswordBearer
- **File Handling**: Python-multipart for file uploads

### Machine Learning Components
- **Feature Extractors**:
  - SIFT (Scale-Invariant Feature Transform): Robust to scaling, rotation, and illumination changes
  - ORB (Oriented FAST and Rotated BRIEF): Fast and efficient local feature detector
  - KAZE: Advanced feature detection with non-linear scale space
  - Vision Transformer (ViT): State-of-the-art deep learning for image understanding
- **Matching Service**: Ensemble approach that combines results from all methods
- **Indexing**: FAISS for fast vector similarity search and clustering

### Infrastructure
- **Frontend Hosting**: Vercel or Netlify
- **Backend Hosting**: Railway, Render, or custom deployment
- **Database**: MongoDB Atlas (cloud) or self-hosted MongoDB
- **Storage**: File system storage for images (with cloud storage option)

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- MongoDB Atlas account or local MongoDB instance
- 4GB+ RAM (for machine learning components)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/tile-matcher.git
   cd tile-matcher
   ```

2. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements-compatible.txt  # Use the compatible requirements file
   ```

3. Set up the frontend:
   ```bash
   cd ../frontend
   npm install
   ```

4. Create test directories for the backend:
   ```bash
   cd ../backend
   mkdir -p test_images/catalog test_images/queries uploads
   ```

### Environment Variables

#### Backend (.env)
```
# Environment
ENVIRONMENT=development
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Authentication
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ALGORITHM=HS256

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=tile_matcher

# File Storage
UPLOAD_FOLDER=./uploads
MAX_FILE_SIZE=5242880  # 5MB
```

#### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret-here
```

## Development

### Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm run dev
```

### Testing the Matching Service
```bash
cd backend
python test_matching.py
```

## API Endpoints

### Authentication
- `POST /token` - Get access token (login)

### Matching
- `POST /api/matching/match` - Upload image and get matching tiles
- `GET /api/matching/methods` - List available matching methods

### Health Check
- `GET /health` - Check API health

## Architecture

### Backend Structure
```
backend/
├── api/
│   ├── endpoints/
│   │   └── matching.py    # API routes for matching
│   └── dependencies.py    # FastAPI dependencies
├── ml/
│   ├── feature_extractors.py  # Image feature extraction algorithms
│   └── matching_service.py    # Tile matching service
├── main.py                # FastAPI app
└── test_matching.py       # Test script
```

### Frontend Structure
```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx       # Home page
│   │   ├── about/         # About page
│   │   ├── login/         # Login page
│   │   └── register/      # Registration page
│   ├── components/
│   │   ├── Header.tsx     # Navigation header
│   │   ├── Footer.tsx     # Page footer
│   │   ├── ImageUpload.tsx # Image upload component
│   │   └── MatchResults.tsx # Results display
│   ├── lib/
│   │   ├── api.ts         # API service
│   │   └── auth.ts        # Authentication utilities
│   └── types/
│       └── index.ts       # TypeScript type definitions
└── public/                # Static assets
```

## Deployment

### Backend Deployment
The backend can be deployed to various platforms:

1. **Railway**:
   - Connect your GitHub repository
   - Select the backend directory as the source
   - Set environment variables in Railway dashboard
   - Deploy with `uvicorn main:app --host 0.0.0.0 --port $PORT`

2. **Render**:
   - Create a new Web Service
   - Connect to your repository
   - Set build command: `pip install -r requirements-compatible.txt`
   - Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables

### Frontend Deployment
1. **Vercel**:
   - Connect your GitHub repository
   - Set the root directory to `/frontend`
   - Add environment variables
   - Deploy

2. **Netlify**:
   - Connect your GitHub repository
   - Set build command: `cd frontend && npm install && npm run build`
   - Set publish directory: `frontend/.next`
   - Add environment variables

## Future Enhancements

- Add CLIP embeddings for semantic understanding
- Implement partial tile matching capability
- Add user roles and permissions
- Enable bulk catalog upload feature
- Create mobile app version
- Add analytics dashboard for matching accuracy

## License

This project is licensed under the MIT License.
