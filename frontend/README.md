# COVID-19 vs Pneumonia Classifier - Frontend

Next.js-based web application for uploading chest X-ray images and receiving AI-powered disease predictions.

## Overview

The frontend provides a user-friendly interface for:
- Uploading chest X-ray images (PNG, JPG, JPEG)
- Viewing real-time prediction results
- Displaying confidence scores for each disease class
- Showing model inference time and metadata

## Requirements

- Node.js 18+
- npm or yarn

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install
```

## Running the Frontend

### Development Mode

```bash
# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Using Docker

```bash
# Build and run with Docker
docker build -t covid-classifier-frontend ./frontend
docker run -p 3000:3000 covid-classifier-frontend
```

### Using Docker Compose

```bash
# Run entire stack (frontend + backend)
docker-compose up --build frontend
```

## Features

### Image Upload
- Drag and drop support
- Click to browse files
- Supports PNG, JPG, JPEG formats
- Image preview before submission

### Prediction Results
- Disease classification: COVID-19, Normal, Pneumonia
- Confidence percentage for each class
- Visual probability distribution
- Model metadata (inference time, model used)

## API Integration

The frontend communicates with the backend API at `http://localhost:8000`

### Prediction Endpoint

```typescript
// POST /api/v1/predict
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

### Response Format

```json
{
  "prediction": "COVID-19",
  "confidence": 0.85,
  "probabilities": {
    "Normal": 0.05,
    "Pneumonia": 0.10,
    "COVID-19": 0.85
  },
  "model_used": "hybrid_cnn_vit",
  "cnn_confidence": 0.82,
  "inference_time": 0.150,
  "cached": false
}
```

## Project Structure

```
frontend/
├── app/                 # Next.js App Router pages
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main page
│   └── globals.css      # Global styles
├── components/          # React components
│   ├── ImageUploader.tsx
│   └── PredictionResult.tsx
├── public/              # Static assets
├── package.json         # Dependencies
├── tailwind.config.js   # Tailwind CSS config
├── tsconfig.json        # TypeScript config
└── next.config.js       # Next.js config
```

## Technology Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Fetch API

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

## Building for Production

```bash
# Install dependencies
npm install

# Create production build
npm run build

# Start optimized production server
npm start
```

## License

MIT License
