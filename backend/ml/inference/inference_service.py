import onnxruntime as ort
import numpy as np
import json
import hashlib
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from app.core.config import settings
from ml.data.preprocessor import XRayPreprocessor

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for loading ONNX models, handling preprocessing, caching, and inference.
    Supports both single and batch predictions with hybrid CNN-ViT architecture.
    
    Features:
    - Optional Redis caching (graceful fallback when unavailable)
    - Fallback mode for demonstration when models are not available
    """

    def __init__(self):
        self.class_names = ['Normal', 'Pneumonia', 'COVID-19']
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Track if we're in fallback mode
        self.fallback_mode = settings.FALLBACK_MODE
        self.models_loaded = False

        # Initialize Redis client (optional)
        self.redis_client = None
        self.redis_enabled = settings.REDIS_ENABLED
        self.cache_ttl = settings.CACHE_TTL
        self._init_redis()

        # Initialize preprocessor
        self.preprocessor = XRayPreprocessor()

        # Load ONNX models (or use fallback)
        self._load_models()

        logger.info("InferenceService initialized successfully")
        if self.fallback_mode:
            logger.warning("Running in FALLBACK mode - using simulated predictions")
        if not self.redis_enabled:
            logger.info("Redis caching disabled")

    def _init_redis(self):
        """Initialize Redis client with graceful fallback"""
        if not self.redis_enabled:
            logger.info("Redis is disabled in configuration")
            return
            
        try:
            import redis
            self.redis_client = redis.from_url(settings.REDIS_URL)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None
            self.redis_enabled = False

    def _load_models(self):
        """Load ONNX models for CNN and ViT, or enable fallback mode"""
        
        # Check if we should use fallback mode
        if self.fallback_mode:
            logger.warning("Fallback mode enabled - models will not be loaded")
            self.cnn_session = None
            self.vit_session = None
            self.models_loaded = False
            return
        
        try:
            # Load CNN model
            cnn_path = Path(settings.CNN_MODEL_PATH)
            if not cnn_path.exists():
                logger.warning(f"CNN model not found at {cnn_path}. Enabling fallback mode.")
                self._enable_fallback_mode()
                return

            self.cnn_session = ort.InferenceSession(
                str(cnn_path),
                providers=['CPUExecutionProvider']
            )
            logger.info(f"Loaded CNN model from {cnn_path}")

            # Load ViT model
            vit_path = Path(settings.VIT_MODEL_PATH)
            if not vit_path.exists():
                logger.warning(f"ViT model not found at {vit_path}. Enabling fallback mode.")
                self._enable_fallback_mode()
                return

            self.vit_session = ort.InferenceSession(
                str(vit_path),
                providers=['CPUExecutionProvider']
            )
            logger.info(f"Loaded ViT model from {vit_path}")
            
            self.models_loaded = True

        except Exception as e:
            logger.error(f"Failed to load models: {e}. Enabling fallback mode.")
            self._enable_fallback_mode()

    def _enable_fallback_mode(self):
        """Enable fallback mode when models are not available"""
        self.fallback_mode = True
        self.cnn_session = None
        self.vit_session = None
        self.models_loaded = False
        logger.warning("FALLBACK MODE ENABLED - Using simulated predictions for demonstration")

    def _get_cache_key(self, image_array: np.ndarray) -> str:
        """Generate cache key from image array"""
        # Create hash of image data
        image_hash = hashlib.md5(image_array.tobytes()).hexdigest()
        return f"inference:{image_hash}"

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction result"""
        if not self.redis_client:
            return None
            
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache prediction result"""
        if not self.redis_client:
            return
            
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single image for inference"""
        # Use validation preprocessing (no augmentations)
        tensor = self.preprocessor.preprocess_single(image, is_training=False)
        # Convert to numpy and add batch dimension (1, C, H, W) for ONNX
        return np.expand_dims(tensor.numpy(), axis=0)

    def _run_cnn_inference(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run CNN inference and return probabilities and confidence"""
        # Get input name
        input_name = self.cnn_session.get_inputs()[0].name

        # Run inference
        outputs = self.cnn_session.run(None, {input_name: input_tensor})

        # Apply softmax
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get confidence (max probability)
        confidence = np.max(probs, axis=1)

        return probs, confidence

    def _run_vit_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ViT inference and return probabilities"""
        # Get input name
        input_name = self.vit_session.get_inputs()[0].name

        # Run inference
        outputs = self.vit_session.run(None, {input_name: input_tensor})

        # Apply softmax
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def _generate_fallback_prediction(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Generate a simulated prediction for demonstration purposes.
        This is used when ONNX models are not available.
        """
        # Use image hash to generate consistent but varied results
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        hash_int = int(image_hash[:8], 16)
        
        # Generate deterministic but varied probabilities
        random.seed(hash_int)
        
        # Create pseudo-random probabilities that vary per image
        base_probs = [random.uniform(0.1, 0.9) for _ in range(3)]
        total = sum(base_probs)
        probs = [p / total for p in base_probs]
        
        # Determine prediction based on highest probability
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Randomly decide if CNN only or hybrid (for demo)
        model_used = 'hybrid' if confidence < 0.85 else 'cnn'
        
        return {
            'prediction': self.class_names[pred_idx],
            'confidence': float(confidence),
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.class_names, probs)
            },
            'model_used': model_used,
            'cnn_confidence': float(random.uniform(0.5, 0.95)),
            'fallback': True
        }

    def predict_single(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform single image prediction with caching.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()

        # Check if we're in fallback mode
        if self.fallback_mode or not self.models_loaded:
            result = self._generate_fallback_prediction(image)
            result['inference_time'] = time.time() - start_time
            result['cached'] = False
            return result

        # Generate cache key
        cache_key = self._get_cache_key(image)

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result['cached'] = True
            logger.info("Cache hit for prediction")
            return cached_result

        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image)

            # Stage 1: CNN prediction
            cnn_probs, cnn_confidences = self._run_cnn_inference(input_tensor)
            cnn_confidence = cnn_confidences[0]

            # Stage 2: Conditional ViT inference
            if cnn_confidence >= self.confidence_threshold:
                final_probs = cnn_probs[0]
                model_used = 'cnn'
            else:
                vit_probs = self._run_vit_inference(input_tensor)[0]

                # Ensemble: weighted average (CNN: 0.4, ViT: 0.6)
                final_probs = 0.4 * cnn_probs[0] + 0.6 * vit_probs
                model_used = 'hybrid'

            final_pred = np.argmax(final_probs)
            final_confidence = np.max(final_probs)

            result = {
                'prediction': self.class_names[final_pred],
                'confidence': float(final_confidence),
                'probabilities': {
                    name: float(prob)
                    for name, prob in zip(self.class_names, final_probs)
                },
                'model_used': model_used,
                'cnn_confidence': float(cnn_confidence),
                'inference_time': time.time() - start_time,
                'cached': False
            }

            # Cache the result
            self._cache_result(cache_key, result)

            logger.info(".3f")
            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}")

    def predict_batch(self, images: List[np.ndarray], batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Perform batch prediction with caching support.

        Args:
            images: List of input images as numpy arrays
            batch_size: Batch size for processing (defaults to settings.BATCH_SIZE)

        Returns:
            List of prediction dictionaries
        """
        # Handle fallback mode for batch
        if self.fallback_mode or not self.models_loaded:
            start_time = time.time()
            results = []
            for image in images:
                result = self._generate_fallback_prediction(image)
                result['inference_time'] = time.time() - start_time
                result['cached'] = False
                results.append(result)
            return results
            
        if batch_size is None:
            batch_size = settings.BATCH_SIZE

        results = []
        start_time = time.time()

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = []

            # Check cache for each image in batch
            uncached_images = []
            uncached_indices = []

            for idx, image in enumerate(batch_images):
                cache_key = self._get_cache_key(image)
                cached_result = self._get_cached_result(cache_key)

                if cached_result:
                    cached_result['cached'] = True
                    batch_results.append((idx, cached_result))
                else:
                    uncached_images.append(image)
                    uncached_indices.append(idx)

            # Process uncached images
            if uncached_images:
                try:
                    # Preprocess batch
                    input_tensors = np.stack([self._preprocess_image(img) for img in uncached_images])

                    # CNN inference for batch
                    cnn_probs_batch, cnn_confidences = self._run_cnn_inference(input_tensors)

                    # Identify images needing ViT
                    vit_indices = [i for i, conf in enumerate(cnn_confidences) if conf < self.confidence_threshold]

                    final_probs_batch = cnn_probs_batch.copy()

                    if vit_indices:
                        # Run ViT for uncertain images
                        vit_input = input_tensors[vit_indices]
                        vit_probs_batch = self._run_vit_inference(vit_input)

                        # Ensemble for uncertain images
                        for local_idx, global_idx in enumerate(vit_indices):
                            final_probs_batch[global_idx] = (
                                0.4 * cnn_probs_batch[global_idx] +
                                0.6 * vit_probs_batch[local_idx]
                            )

                    # Create results for uncached images
                    for batch_idx, (global_idx, image) in enumerate(zip(uncached_indices, uncached_images)):
                        final_probs = final_probs_batch[batch_idx]
                        cnn_conf = cnn_confidences[batch_idx]

                        final_pred = np.argmax(final_probs)
                        final_confidence = np.max(final_probs)

                        model_used = 'cnn' if cnn_conf >= self.confidence_threshold else 'hybrid'

                        result = {
                            'prediction': self.class_names[final_pred],
                            'confidence': float(final_confidence),
                            'probabilities': {
                                name: float(prob)
                                for name, prob in zip(self.class_names, final_probs)
                            },
                            'model_used': model_used,
                            'cnn_confidence': float(cnn_conf),
                            'inference_time': time.time() - start_time,
                            'cached': False
                        }

                        # Cache result
                        cache_key = self._get_cache_key(image)
                        self._cache_result(cache_key, result)

                        batch_results.append((global_idx, result))

                except Exception as e:
                    logger.error(f"Batch inference failed: {e}")
                    # Return error results for failed batch
                    for idx in uncached_indices:
                        results.append({
                            'error': f'Inference failed: {str(e)}',
                            'cached': False
                        })

            # Sort results by original order and add to main results
            batch_results.sort(key=lambda x: x[0])
            results.extend([result for _, result in batch_results])

        total_time = time.time() - start_time
        logger.info(f"Batch prediction completed for {len(images)} images in {total_time:.3f}s")

        return results

    def clear_cache(self):
        """Clear all cached predictions"""
        if not self.redis_client:
            return {'message': 'Cache disabled', 'cached_predictions': 0}
            
        try:
            keys = self.redis_client.keys("inference:*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached predictions")
            return {'cleared_predictions': len(keys)}
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return {'error': str(e)}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {
                'cached_predictions': 0,
                'cache_ttl': self.cache_ttl,
                'redis_enabled': False,
                'fallback_mode': self.fallback_mode
            }
            
        try:
            keys = self.redis_client.keys("inference:*")
            return {
                'cached_predictions': len(keys),
                'cache_ttl': self.cache_ttl,
                'redis_enabled': True,
                'fallback_mode': self.fallback_mode
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}