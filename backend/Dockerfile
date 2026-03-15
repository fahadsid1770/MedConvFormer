# Use Python 3.12 slim
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled PyTorch (use cu121 for CUDA 12.1)
# If building for CPU-only, use: torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for caching
COPY requirements.txt .

# Install remaining Python dependencies (except torch/torchvision which are already installed)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torch torchvision 2>/dev/null || true

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
