'use client';

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
  fallback?: boolean;
}

interface PredictionResultProps {
  prediction: PredictionResponse;
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
  const getPredictionColor = (pred: string) => {
    switch (pred) {
      case 'COVID-19':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'Pneumonia':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'Normal':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getConfidenceLevel = (conf: number) => {
    if (conf >= 0.8) return { label: 'High', color: 'text-green-600' };
    if (conf >= 0.5) return { label: 'Medium', color: 'text-yellow-600' };
    return { label: 'Low', color: 'text-red-600' };
  };

  const confidenceInfo = getConfidenceLevel(prediction.confidence);

  return (
    <div className="space-y-6">
      {/* Fallback Mode Warning */}
      {prediction.fallback && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded text-sm">
          ⚠️ Running in demo mode - predictions are simulated
        </div>
      )}

      {/* Main Prediction */}
      <div className="text-center">
        <div className={`inline-block px-6 py-3 rounded-lg border-2 ${getPredictionColor(prediction.prediction)}`}>
          <span className="text-2xl font-bold">{prediction.prediction}</span>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-gray-600 font-medium">Confidence Score</span>
          <span className={`font-bold ${confidenceInfo.color}`}>
            {confidenceInfo.label} ({prediction.confidence.toFixed(2)})
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div
            className="bg-indigo-600 h-4 rounded-full transition-all"
            style={{ width: `${prediction.confidence * 100}%` }}
          />
        </div>
        <p className="text-right text-sm text-gray-500">
          {(prediction.confidence * 100).toFixed(1)}%
        </p>
      </div>

      {/* Probability Distribution */}
      <div className="space-y-3">
        <h4 className="font-medium text-gray-700">Class Probabilities</h4>
        {Object.entries(prediction.probabilities).map(([className, prob]) => (
          <div key={className} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">{className}</span>
              <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  className === 'COVID-19' ? 'bg-red-500' :
                  className === 'Pneumonia' ? 'bg-orange-500' : 'bg-green-500'
                }`}
                style={{ width: `${prob * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Model Info */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="bg-gray-50 rounded p-3">
          <span className="text-gray-500 block">Model Used</span>
          <span className="font-medium capitalize">{prediction.model_used}</span>
        </div>
        <div className="bg-gray-50 rounded p-3">
          <span className="text-gray-500 block">Inference Time</span>
          <span className="font-medium">{prediction.inference_time.toFixed(3)}s</span>
        </div>
        <div className="bg-gray-50 rounded p-3">
          <span className="text-gray-500 block">CNN Confidence</span>
          <span className="font-medium">{(prediction.cnn_confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="bg-gray-50 rounded p-3">
          <span className="text-gray-500 block">Cached Result</span>
          <span className="font-medium">{prediction.cached ? 'Yes' : 'No'}</span>
        </div>
      </div>
    </div>
  );
}
