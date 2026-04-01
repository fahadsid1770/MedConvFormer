# COVID-19 vs Pneumonia Classifier - Backend

FastAPI-based backend service for COVID-19 and Pneumonia detection from chest X-ray images using hybrid CNN-Transformer architecture.

## Overview

The backend provides RESTful API endpoints for:
- Health check monitoring
- Image-based disease prediction
- Model inference with confidence scores
- Model optimization and export utilities

## Requirements

- Python 3.9+
- PyTorch 2.0+
- FastAPI
- CUDA-capable GPU (optional, for faster inference)

See [`requirements.txt`](requirements.txt) for full dependencies.

## Installation

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Important Commands

### Training Models

Train a CNN model:
```bash
python3 ml/training/train_model.py --model cnn --epochs 10
```

Train a ViT model:
```bash
python3 ml/training/train_model.py --model vit --epochs 10
```

Train and export to ONNX:
```bash
python3 ml/training/train_model.py --model cnn --epochs 10 --export
```

Training options:
- `--model`: Model type (`cnn` or `vit`)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--img-size`: Image size (default: 224)
- `--export`: Export to ONNX after training

### Model Optimization

Export model to ONNX format:
```bash
python3 -m ml.optimization.cli export --model-type cnn --cnn-path models/cnn_model.pth --output-dir exports
```

Quantize model:
```bash
python3 -m ml.optimization.cli quantize --model-type cnn --model-path models/cnn_model.pth --quantization-type dynamic
```

Benchmark models:
```bash
python3 -m ml.optimization.cli benchmark --model-type cnn --model-path models/cnn_model.pth
```

## Running the Backend

### Development Mode

```bash
# Run with uvicorn
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker

```bash
# Build and run with Docker
docker build -t covid-classifier-backend ./backend
docker run -p 8000:8000 -v ./models:/app/models -v ./data:/app/data covid-classifier-backend
```

### Using Docker Compose

```bash
# Run entire stack (backend + frontend)
docker-compose up --build backend
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check endpoint |
| `/api/v1/predict` | POST | Submit X-ray image for prediction |

### Prediction Request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@xray_image.jpg"
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
backend/
├── app/
│   ├── api/           # API route handlers
│   ├── core/          # Configuration
│   ├── services/     # Business logic
│   └── utils/         # Utilities
├── ml/
│   ├── inference/     # Model inference
│   ├── optimization/ # Model optimization & export
│   └── training/     # Model training scripts
├── models/           # Trained model files
├── tests/            # Test suite
├── run.py            # Application entry point
└── requirements.txt  # Python dependencies
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `DATA_ROOT` | Path to dataset | `./data` |
| `MODEL_DIR` | Path to models | `./models` |
| `FALLBACK_MODE` | Enable fallback mode | `false` |
| `REDIS_ENABLED` | Enable Redis caching | `false` |

## Testing

```bash
# Run all tests
pytest backend/tests/

# Run specific test file
pytest backend/tests/test_api_predict.py -v
```

## License

MIT License
