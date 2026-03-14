#!/usr/bin/env python3
"""
Run script for the COVID-19 Detection API backend

Usage:
    python3 run.py
    
Or with custom settings:
    FALLBACK_MODE=true python3 run.py  # Use fallback mode (no models required)
    REDIS_ENABLED=true python3 run.py  # Enable Redis caching
"""
import os
import sys

# Ensure the backend directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )