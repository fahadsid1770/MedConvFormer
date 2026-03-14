'use client';

import { useState, useCallback } from 'react';
import ImageUploader from '@/components/ImageUploader';
import PredictionResult from '@/components/PredictionResult';

interface PredictionResponse {
  prediction: string;
  confidence: number;
  probabilities: {
    Normal: number;
    Pneumonia: number;
    'COVID-19': number;
  };
  model_used: string;
  cnn_confidence: number;
  inference_time: number;
  cached: boolean;
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = useCallback((imageDataUrl: string) => {
    setSelectedImage(imageDataUrl);
    setPrediction(null);
    setError(null);
  }, []);

  const handlePredict = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);

    try {
      // Convert base64 to blob
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      
      // Create form data
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');

      // Make prediction request
      const result = await fetch('http://localhost:8000/api/v1/predict/single', {
        method: 'POST',
        body: formData,
      });

      if (!result.ok) {
        throw new Error(`Prediction failed: ${result.statusText}`);
      }

      const data = await result.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center">
        <h2 className="text-4xl font-extrabold text-gray-900 sm:text-5xl">
          Upload an X-ray Image
        </h2>
        <p className="mt-4 text-xl text-gray-600">
          Our AI-powered hybrid CNN-Transformer model will analyze your chest X-ray
          and detect COVID-19, Pneumonia, or Normal conditions.
        </p>
      </div>

      {/* Upload Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Upload X-ray Image</h3>
          <ImageUploader onImageSelect={handleImageSelect} />
          
          {selectedImage && (
            <div className="mt-4">
              <button
                onClick={handlePredict}
                disabled={isLoading}
                className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg font-medium 
                         hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed
                         transition-colors"
              >
                {isLoading ? 'Analyzing...' : 'Analyze X-ray'}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
          
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
              {error}
            </div>
          )}
          
          {prediction ? (
            <PredictionResult prediction={prediction} />
          ) : (
            <div className="text-center text-gray-500 py-12">
              {selectedImage 
                ? 'Click "Analyze X-ray" to get prediction'
                : 'Upload an image to see results'
              }
            </div>
          )}
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-3xl mb-2">🔬</div>
          <h4 className="font-semibold text-lg">Hybrid AI Model</h4>
          <p className="text-gray-600 text-sm mt-2">
            Combines EfficientNet CNN for fast inference with Vision Transformer for 
            enhanced accuracy on uncertain predictions.
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-3xl mb-2">⚡</div>
          <h4 className="font-semibold text-lg">Two-Stage Inference</h4>
          <p className="text-gray-600 text-sm mt-2">
            CNN provides quick predictions. ViT is used only when confidence is below 
            threshold for improved accuracy.
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-3xl mb-2">🎯</div>
          <h4 className="font-semibold text-lg">High Accuracy</h4>
          <p className="text-gray-600 text-sm mt-2">
            Trained on COVID-19 Radiography Database with thousands of X-ray images 
            for reliable detection.
          </p>
        </div>
      </div>
    </div>
  );
}
