# MedConvFormer: Hybrid CNN-Transformer DL system to Distinguish COVID-19, Normal, and Pneumonia from chest X-ray images

A hybrid CNN-Transformer deep learning system for distinguishing COVID-19, Normal, and Pneumonia from chest X-ray images.

---

What Makes This Project Stand Out

Innovative Hybrid Architecture

This project implements a **novel hybrid deep learning architecture** that uniquely combines the strengths of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViT)** for medical image classification. Unlike traditional single-model approaches, this system leverages:

- **EfficientNet-B0** for robust local feature extraction with computational efficiency
- **Vision Transformer (ViT)** for capturing global context and long-range dependencies in chest X-rays
- **Adaptive Ensemble Inference** that dynamically selects between CNN-only and hybrid CNN+ViT mode based on prediction confidence

Key Exceptional Features

1. **Intelligent Model Switching**: The system automatically determines whether to use CNN-only inference (when confidence is high ≥85%) or trigger the hybrid ensemble mode (when confidence is low <85%) for more accurate predictions.

2. **Production-Ready ML Pipeline**: End-to-end machine learning pipeline with:
   - Model training with customizable hyperparameters
   - **ONNX export** for cross-platform deployment
   - **Model quantization** for optimized inference
   - **Performance benchmarking** tools

3. **Advanced Caching**:
   - Redis-powered inference caching

5. **Comprehensive Data Augmentation**: Training pipeline includes:
   - Random horizontal flips
   - Random rotation (±15°)
   - Color jitter (brightness/contrast)
   - ImageNet normalization

---

Model Architecture

### Hybrid CNN-Transformer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Chest X-ray                       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                   ┌───────────────────┐
│   EfficientNet-B0 │                   │  Vision Transformer│
│      (CNN)        │                   │  (ViT Small)      │
│                   │                   │                   │
│ - Local features  │                   │ - Global context  │
│ - Efficient       │                   │ - Long-range dep  │
└───────────────────┘                   └───────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ▼
                 ┌────────────────────────┐
                 │   Adaptive Ensemble    │
                 │   Weight: CNN 0.4      │
                 │          + ViT 0.6     │
                 └────────────────────────┘
                              │
                              ▼
              ┌────────────────────────────┐
              │  OUTPUT: Class Prediction  │
              │  - COVID-19 (0)            │
              │  - Normal (1)              │
              │  - Pneumonia (2)           │
              └────────────────────────────┘
```

### Technical Specifications

| Component | Model | Details |
|-----------|-------|---------|
| CNN Backbone | EfficientNet-B0 | Pretrained on ImageNet, custom classifier head |
| Transformer | ViT Small patch16 | 224×224 input, pretrained |
| Classifier | Custom Head | Dropout → Linear → GELU → Linear |
| Ensemble | Weighted Average | CNN: 40%, ViT: 60% weights |

---

Dataset

The model is trained on the **COVID-19 Radiography Database** - a standardized dataset of chest X-ray images:

| Class | Images | Description |
|-------|--------|-------------|
| **COVID-19** | ~3,000 | SARS-CoV-2 infected pneumonia |
| **Normal** | ~10,000 | Healthy lung X-ray |
| **Viral Pneumonia** | ~1,500 | Viral pneumonia |

**Total**: ~14,500 chest X-ray images

Dataset location: `data/COVID-19_Radiography_Dataset/`

---

Training Details

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 5 (default) | Number of training epochs |
| `batch-size` | 8 | Training batch size |
| `learning-rate` | 1e-4 | AdamW optimizer learning rate |
| `image-size` | 224×224 | Input image dimensions |
| `weight-decay` | 0.01 | L2 regularization |
| `train-split` | 80% | Training data ratio |
| `val-split` | 20% | Validation data ratio |

### Training Pipeline

1. **Data Preprocessing**: Resize → Augmentation → Normalization (ImageNet stats)
2. **Optimizer**: AdamW with CosineAnnealingLR scheduler
3. **Loss Function**: CrossEntropyLoss
4. **Early Stopping**: Best model saved based on validation accuracy
5. **Model Export**: Optional ONNX export after training

---

Evaluation Metrics

The training and validation process tracks:

- **Accuracy**: Percentage of correct predictions
- **Loss**: Cross-entropy loss value
- **Precision/Recall/F1**: Per-class metrics (computed during evaluation)
- **Confusion Matrix**: For detailed error analysis

### Expected Performance

Based on the hybrid architecture and dataset:
- **Validation Accuracy**: ~90-95% (depending on epochs)
- **COVID-19 Detection**: High sensitivity due to distinctive patterns
- **Normal vs Pneumonia**: Robust differentiation with ViT global context

---

Quick Start

### Training Commands

Train CNN model (10 epochs, export to ONNX):
```bash
python3 backend/ml/training/train_model.py --model cnn --epochs 10 --export
```

Train ViT model:
```bash
python3 backend/ml/training/train_model.py --model vit --epochs 10
```

#### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model type (`cnn` or `vit`) | `cnn` |
| `--epochs` | Number of training epochs | `5` |
| `--batch-size` | Batch size | `8` |
| `--lr` | Learning rate | `1e-4` |
| `--img-size` | Image size | `224` |
| `--export` | Export to ONNX after training | `false` |

### Model Optimization Commands

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

### Running the Application

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

---

Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── app/                 # API application
│   ├── ml/                 # Machine learning modules
│   │   ├── inference/      # Inference service (hybrid ensemble)
│   │   ├── optimization/   # Model optimization (ONNX, quantization)
│   │   └── training/      # Training scripts
│   ├── models/             # Trained model files
│   ├── tests/              # Test suite
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app pages
│   ├── components/         # React components
│   └── package.json       # Node.js dependencies
├── data/                  # COVID-19 Radiography Dataset
├── docker-compose.yml     # Docker orchestration
└── README.md              # This file
```

---

Technology Stack

### Backend
- **Framework**: FastAPI
- **ML Framework**: PyTorch 2.0+
- **Model Zoo**: TIMM (PyTorch Image Models)
- **Inference Runtime**: ONNX Runtime
- **Caching**: Redis
- **Deployment**: Docker, Uvicorn

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Deployment**: Docker, Node.js

---

API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict` | POST | Submit X-ray for prediction |

---

Requirements

### Backend
- Python 3.9+
- CUDA-capable GPU (optional, recommended for training)

### Frontend
- Node.js 18+
- npm or yarn

---

License

MIT License
