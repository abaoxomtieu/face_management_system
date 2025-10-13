import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from pathlib import Path
import io
from loguru import logger


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load image from bytes
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        Image as numpy array (RGB format) or None if failed
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        return None


def load_image_from_file(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (RGB format) or None if failed
    """
    try:
        # Read image with OpenCV (BGR format)
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to read image from {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        logger.error(f"Failed to load image from file: {e}")
        return None


def save_image(image, save_path: Path) -> bool:
    """
    Save image to file
    
    Args:
        image: Image as numpy array (RGB format) or bytes
        save_path: Path to save image
        
    Returns:
        True if successful
    """
    try:
        # Create parent directory if not exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If image is bytes, convert to numpy array first
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to decode image from bytes")
                return False
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            logger.error(f"Image is not numpy array, got {type(image)}")
            return False
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(str(save_path), image_bgr)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


def resize_image(
    image: np.ndarray,
    max_size: int = 1920,
    maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image to maximum size while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if maintain_aspect_ratio:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
    else:
        new_height = new_width = max_size
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized


def check_image_quality(image: np.ndarray) -> Tuple[bool, str, float]:
    """
    Check image quality (blur detection, brightness, etc.)
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (is_good_quality, message, quality_score)
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Check blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 100.0
        
        if laplacian_var < blur_threshold:
            return False, f"Image is too blurry (score: {laplacian_var:.2f})", laplacian_var / blur_threshold
        
        # Check brightness
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 40:
            return False, f"Image is too dark (brightness: {mean_brightness:.2f})", mean_brightness / 255.0
        
        if mean_brightness > 220:
            return False, f"Image is too bright (brightness: {mean_brightness:.2f})", mean_brightness / 255.0
        
        # Calculate overall quality score (0-1)
        blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize to 0-1
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5  # Optimal at 127.5
        
        quality_score = (blur_score * 0.6 + brightness_score * 0.4)
        
        return True, "Good quality", quality_score
        
    except Exception as e:
        logger.error(f"Failed to check image quality: {e}")
        return False, f"Error checking quality: {e}", 0.0


def draw_face_boxes(
    image: np.ndarray,
    faces: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes around detected faces
    
    Args:
        image: Input image
        faces: List of face detection results with 'bbox' key
        color: Box color (RGB)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    image_copy = image.copy()
    
    for face in faces:
        bbox = face.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label if available
            label = face.get('name', face.get('student_id', ''))
            if label:
                cv2.putText(
                    image_copy,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness
                )
    
    return image_copy


def crop_face(image: np.ndarray, bbox: List[float], margin: float = 0.2) -> Optional[np.ndarray]:
    """
    Crop face from image with margin
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Margin ratio to add around face
        
    Returns:
        Cropped face image or None
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Calculate new coordinates with margin
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop
    except Exception as e:
        logger.error(f"Failed to crop face: {e}")
        return None


def calculate_face_area(bbox: List[float]) -> float:
    """Calculate face area from bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def is_face_too_small(bbox: List[float], image_shape: Tuple[int, int], min_ratio: float = 0.01) -> bool:
    """
    Check if face is too small relative to image
    
    Args:
        bbox: Face bounding box
        image_shape: Image shape (height, width)
        min_ratio: Minimum face area ratio to image area
        
    Returns:
        True if face is too small
    """
    face_area = calculate_face_area(bbox)
    image_area = image_shape[0] * image_shape[1]
    
    return (face_area / image_area) < min_ratio


def validate_image_format(image_bytes: bytes, allowed_formats: List[str] = None) -> Tuple[bool, str]:
    """
    Validate image format
    
    Args:
        image_bytes: Image bytes
        allowed_formats: List of allowed formats (e.g., ['JPEG', 'PNG'])
        
    Returns:
        Tuple of (is_valid, format_or_error_message)
    """
    if allowed_formats is None:
        allowed_formats = ['JPEG', 'PNG', 'JPG', 'BMP', 'WEBP']
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format
        
        if image_format not in allowed_formats:
            return False, f"Unsupported format: {image_format}. Allowed: {allowed_formats}"
        
        return True, image_format
    except Exception as e:
        return False, f"Invalid image: {e}"


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length (L2 normalization)
    
    Args:
        embedding: Face embedding vector
        
    Returns:
        Normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score (0-1)
    """
    # Normalize embeddings
    emb1_norm = normalize_embedding(embedding1)
    emb2_norm = normalize_embedding(embedding2)
    
    # Calculate dot product (cosine similarity for normalized vectors)
    similarity = np.dot(emb1_norm, emb2_norm)
    
    return float(similarity)


def generate_session_id(class_name: Optional[str] = None) -> str:
    """
    Generate attendance session ID
    
    Args:
        class_name: Optional class name to include in session ID
        
    Returns:
        Session ID string
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if class_name:
        return f"SESSION_{class_name}_{timestamp}"
    else:
        return f"SESSION_{timestamp}"


def batch_process_images(
    image_paths: List[Path],
    process_func,
    batch_size: int = 10
) -> List:
    """
    Process images in batches
    
    Args:
        image_paths: List of image paths
        process_func: Function to process each image
        batch_size: Batch size
        
    Returns:
        List of processing results
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        for image_path in batch:
            try:
                result = process_func(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(None)
    
    return results


def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """
    Create thumbnail of image
    
    Args:
        image: Input image
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return thumbnail

