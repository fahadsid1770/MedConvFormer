# COVID-19 vs Pneumonia Classifier

A hybrid CNN-Transformer deep learning system for distinguishing COVID-19, Normal, and Pneumonia from chest X-ray images.

## Description

This project implements a hybrid deep learning architecture combining Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for medical image classification. The system is designed to accurately classify chest X-ray images into three categories:

- **COVID-19**: SARS-CoV-2 infected pneumonia
- **Normal**: Healthy lung X-ray
- **Pneumonia**: Viral pneumonia

The hybrid approach leverages the local feature extraction capabilities of CNNs and the global context understanding of Transformers for improved diagnostic accuracy.

## Quick Start

### Important Commands

#### Training Models

Train CNN model (10 epochs, export to ONNX):
```bash
python3 backend/ml/training/train_model.py --model cnn --epochs 10 --export
```

Train ViT model:
```bash
python3 backend/ml/training/train_model.py --model vit --epochs 10
```

Training options:
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model type (`cnn` or `vit`) | `cnn` |
| `--epochs` | Number of training epochs | `5` |
| `--batch-size` | Batch size | `8` |
| `--lr` | Learning rate | `1e-4` |
| `--img-size` | Image size | `224` |
| `--export` | Export to ONNX after training | `false` |

#### Model Optimization

Export to ONNX:
```bash
python3 -m ml.optimization.cli export --model-type cnn --cnn-path models/cnn_model.pth --output-dir exports
```

Quantize model:
```bash
python3 -m ml.optimization.cli quantize --model-type cnn --model-path models/cnn_model.pth --quantization-type dynamic
```

Benchmark model:
```bash
python3 -m ml.optimization.cli benchmark --model-type cnn --model-path models/cnn_model.pth
```

#### Running the Application

Using Docker Compose (recommended):
```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d
```

Services will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

Manual setup:
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── app/                 # API application
│   ├── ml/                 # Machine learning modules
│   │   ├── inference/      # Inference service
│   │   ├── optimization/   # Model optimization
│   │   └── training/      # Training scripts
│   ├── models/             # Trained model files
│   ├── tests/              # Test suite
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app pages
│   ├── components/         # React components
│   └── package.json       # Node.js dependencies
├── docker-compose.yml     # Docker orchestration
└── README.md              # This file
```

## Technology Stack

### Backend
- **Framework**: FastAPI
- **ML Framework**: PyTorch 2.0+
- **Model Zoo**: TIMM (PyTorch Image Models)
- **Deployment**: Docker, Uvicorn

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Deployment**: Docker, Node.js

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict` | POST | Submit X-ray for prediction |

## Dataset

The model is trained on the COVID-19 Radiography Database:
- COVID-19: ~3,000 images
- Normal: ~10,000 images  
- Viral Pneumonia: ~1,500 images

Place dataset in: `data/COVID-19_Radiography_Dataset/`

## Requirements

### Backend
- Python 3.9+
- CUDA-capable GPU (optional)

### Frontend
- Node.js 18+
- npm or yarn

## License

MIT License
