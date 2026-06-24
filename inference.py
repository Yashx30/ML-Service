# ml/app/services/inference.py
import os
import logging
from typing import Dict, List, Optional
from io import BytesIO
import tempfile
import requests
from PIL import Image
import boto3
import json
import base64
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- SageMaker Configuration from ENV ----
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "tooth-disease-serverless")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# ---- Class mapping (must match training) ----
ID2LABEL = {
    0: "Calculus",
    1: "Caries",
    2: "Gingivitis",
    3: "Mouth Ulcer",
    4: "Tooth Discoloration",
    5: "Hypodontia",
}

# Lazy-loaded SageMaker client
_sagemaker_runtime = None

def _get_sagemaker_client():
    """Get or create SageMaker runtime client."""
    global _sagemaker_runtime
    
    if _sagemaker_runtime is None:
        logger.info(f"[ML][SageMaker] Initializing client for endpoint: {SAGEMAKER_ENDPOINT_NAME}")
        
        # Create client with credentials from env vars
        session_kwargs = {'region_name': AWS_REGION}
        
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            session_kwargs['aws_access_key_id'] = AWS_ACCESS_KEY_ID
            session_kwargs['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY
        
        _sagemaker_runtime = boto3.client('sagemaker-runtime', **session_kwargs)
        logger.info(f"[ML][SageMaker] Client initialized for region: {AWS_REGION}")
    
    return _sagemaker_runtime

def _detect_and_crop_mouth(image: Image.Image) -> Image.Image:
    """
    Detect face and crop mouth region using OpenCV.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image of cropped mouth region, or original if face/mouth not detected
    """
    try:
        logger.info("[ML][FaceDetection] Detecting face and mouth region...")
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            logger.info("[ML][FaceDetection] No face detected - using original image")
            return image
        
        logger.info(f"[ML][FaceDetection] Detected {len(faces)} face(s)")
        
        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        (fx, fy, fw, fh) = face
        
        logger.info(f"[ML][FaceDetection] Face region: ({fx}, {fy}, {fw}, {fh})")
        
        # Define mouth region (lower 40% of face, centered)
        mouth_y_start = int(fy + fh * 0.6)  # Start at 60% down
        mouth_y_end = int(fy + fh)          # End at bottom
        mouth_x_start = int(fx + fw * 0.2)  # 20% from left
        mouth_x_end = int(fx + fw * 0.8)    # 20% from right
        
        # Add padding for better context
        padding = int(fh * 0.1)
        mouth_y_start = max(0, mouth_y_start - padding)
        mouth_y_end = min(img_bgr.shape[0], mouth_y_end + padding)
        mouth_x_start = max(0, mouth_x_start - padding)
        mouth_x_end = min(img_bgr.shape[1], mouth_x_end + padding)
        
        # Crop mouth region
        mouth_region = img_bgr[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        
        if mouth_region.size == 0:
            logger.warning("[ML][FaceDetection] Mouth crop failed - using original")
            return image
        
        logger.info(f"[ML][FaceDetection] Cropped mouth: {mouth_region.shape[1]}x{mouth_region.shape[0]} pixels")
        
        # Convert back to RGB PIL Image
        mouth_rgb = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB)
        mouth_pil = Image.fromarray(mouth_rgb)
        
        return mouth_pil
        
    except Exception as e:
        logger.warning(f"[ML][FaceDetection] Detection failed: {e} - using original image")
        return image

def _url_to_image(url: str) -> Image.Image:
    """Download image from URL."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to fetch/parse image URL '{url}': {e}")

def _call_sagemaker_predict(image: Image.Image) -> Dict[str, float]:
    """
    Call SageMaker endpoint for inference with automatic mouth detection.

    Args:
        image: PIL Image (can be full face or close-up)

    Returns:
        Dictionary mapping class labels to probabilities
    """
    try:
        # Automatically detect and crop mouth if it's a face photo
        processed_image = _detect_and_crop_mouth(image)
        
        logger.info("[ML][SageMaker] Preparing image for inference...")
        
        # Convert image to base64
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create payload
        payload = {
            'image': image_base64
        }
        
        logger.info(f"[ML][SageMaker] Calling endpoint: {SAGEMAKER_ENDPOINT_NAME}")
        
        # Invoke endpoint
        client = _get_sagemaker_client()
        response = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        response_body = response['Body'].read().decode()
        result = json.loads(response_body)
        
        logger.info(f"[ML][SageMaker] Raw API response: {result}")

        # Validate response format
        if not isinstance(result, dict):
            logger.error(f"[ML][SageMaker] Unexpected response type: {type(result)}")
            return {label: 0.0 for label in ID2LABEL.values()}
        
        # Ensure all classes exist with proper rounding
        parsed_result = {}
        for label in ID2LABEL.values():
            parsed_result[label] = round(float(result.get(label, 0.0)), 6)

        logger.info("[ML][SageMaker] Parsed class probabilities:")
        for label, score in parsed_result.items():
            logger.info(f"  - {label}: {score}")

        return parsed_result

    except Exception as e:
        logger.error(f"[ML][SageMaker] API call failed: {e}")
        raise

# ---- PUBLIC API ----
def infer_image(url: str) -> Dict[str, float]:
    """
    Runs inference on a single remote image URL using SageMaker endpoint.
    Automatically detects faces and crops mouth region if applicable.

    Args:
        url: HTTP URL to image

    Returns:
        Dictionary mapping class labels to probabilities
    """
    logger.info(f"[ML][infer_image] Processing URL via SageMaker: {url[:120]}...")
    image = _url_to_image(url)
    return _call_sagemaker_predict(image)

def aggregate_views(urls: List[str]) -> Dict[str, float]:
    """
    Multi-image inference — returns max confidence per class.
    Automatically detects faces and crops mouth regions.

    Args:
        urls: list of image URLs

    Returns:
        Dict[str, float]: max confidence per class across images
    """
    if not urls:
        raise ValueError("No image URLs provided")

    logger.info("[ML][aggregate_views] Aggregating across URLs using SageMaker:")
    for u in urls:
        logger.info(f"  - {u[:160]}")

    agg: Dict[str, float] = {label: 0.0 for label in ID2LABEL.values()}

    for url in urls[:5]:  # safety limit
        logger.info(f"[ML][aggregate_views] Running infer_image for URL: {url[:160]}")
        try:
            res = infer_image(url)
            logger.info("[ML][aggregate_views] Per-image result:")
            for label, score in res.items():
                logger.info(f"  - {label}: {score}")
            for label, score in res.items():
                agg[label] = max(agg[label], score)
        except Exception as e:
            logger.warning(f"[ML][aggregate_views][WARN] Skipping failed image '{url}': {e}")

    logger.info("[ML][aggregate_views] Final aggregated max confidences per class:")
    for label, score in agg.items():
        logger.info(f"  - {label}: {score}")

    return agg
