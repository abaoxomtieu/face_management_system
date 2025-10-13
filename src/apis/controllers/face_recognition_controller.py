import numpy as np
import threading
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time
from loguru import logger

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    logger.error("InsightFace not installed. Please install: pip install insightface")
    raise

from src.utils.faiss_manager import get_faiss_manager
from src.utils.class_manager import get_class_manager
from src.utils.face_recognition_utils import (
    load_image_from_bytes,
    save_image,
    check_image_quality,
    is_face_too_small
)
from src.config import settings


class FaceRecognitionController:
    """
    Optimized Face Recognition Controller using InsightFace (ArcFace)
    Singleton pattern for model reuse and performance optimization
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = None,
        detection_threshold: float = None,
        recognition_threshold: float = None
    ):
        """Initialize the face recognition controller"""
        if hasattr(self, '_initialized'):
            return
            
        self.model_name = model_name or settings.INSIGHTFACE_MODEL
        self.detection_threshold = detection_threshold or settings.FACE_DETECTION_THRESHOLD
        self.recognition_threshold = recognition_threshold or settings.FACE_RECOGNITION_THRESHOLD
        
        # Initialize components
        self.faiss_manager = get_faiss_manager()
        self.class_manager = get_class_manager()
        
        # Load model
        self._load_model()
        
        # Load existing data
        self._load_data()
        
        self._initialized = True
        logger.info("FaceRecognitionController initialized successfully")
    
    def _load_model(self):
        """Load InsightFace model"""
        try:
            logger.info(f"Loading InsightFace model: {self.model_name}")
            self.app = FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.model_loaded = True
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            self.model_loaded = False
            raise
    
    def _load_data(self):
        """Load existing data"""
        try:
            # Load Faiss index
            self.faiss_manager.load_index()
            logger.info("Faiss index loaded successfully")
            
            # Load class data
            self.class_manager.load_classes()
            logger.info("Class data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")
    
    def register_student(
        self, 
        student_id: str, 
        class_id: str, 
        images: List[bytes]
    ) -> Tuple[bool, str, Dict]:
        """
        Register a new student with face images
        
        Args:
            student_id: Unique student ID
            class_id: Class ID
            images: List of image bytes
            
        Returns:
            (success, message, student_info)
        """
        try:
            start_time = time.time()
            
            # Validate class exists
            if not self.class_manager.class_exists(class_id):
                return False, f"Class {class_id} does not exist", {}
            
            # Process images and extract embeddings
            embeddings = []
            valid_images = []
            
            for i, image_bytes in enumerate(images):
                try:
                    # Load and validate image
                    image = load_image_from_bytes(image_bytes)
                    if not check_image_quality(image):
                        logger.warning(f"Image {i+1} quality check failed for student {student_id}")
                        continue
                    
                    # Detect faces
                    faces = self.app.get(image)
                    if not faces:
                        logger.warning(f"No faces detected in image {i+1} for student {student_id}")
                        continue
                    
                    # Use the largest face
                    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    
                    # Check face size
                    if is_face_too_small(largest_face.bbox, image.shape):
                        logger.warning(f"Face too small in image {i+1} for student {student_id}")
                        continue
                    
                    # Extract embedding
                    embedding = largest_face.embedding
                    embeddings.append(embedding)
                    valid_images.append(image_bytes)
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {i+1} for student {student_id}: {e}")
                    continue
            
            if not embeddings:
                return False, "No valid faces found in provided images", {}
            
            # Save images
            student_dir = settings.STUDENT_IMAGES_DIR / student_id
            student_dir.mkdir(parents=True, exist_ok=True)
            
            for i, image_bytes in enumerate(valid_images):
                image_path = student_dir / f"image_{i+1}.jpg"
                save_image(image_bytes, image_path)
            
            # Add to Faiss index
            embeddings_array = np.array(embeddings)
            self.faiss_manager.add_embeddings(
                embeddings=embeddings_array,
                student_id=student_id,
                class_id=class_id
            )
            
            # Update class student count
            self.class_manager.increment_student_count(class_id)
            
            # Save data
            self.faiss_manager.save_index()
            self.class_manager.save_classes()
            
            processing_time = time.time() - start_time
            
            student_info = {
                "student_id": student_id,
                "class_id": class_id,
                "num_images": len(valid_images),
                "created_at": time.time(),
                "processing_time": processing_time
            }
            
            logger.info(f"Student {student_id} registered successfully with {len(valid_images)} images in {processing_time:.2f}s")
            return True, "Student registered successfully", student_info
            
        except Exception as e:
            logger.error(f"Student registration failed: {e}")
            return False, f"Registration failed: {str(e)}", {}
    
    def get_all_students(self, class_id: Optional[str] = None) -> Tuple[bool, str, List[Dict]]:
        """
        Get all students with optional class filter
        
        Args:
            class_id: Optional class filter
            
        Returns:
            (success, message, students)
        """
        try:
            students = self.faiss_manager.get_all_students()
            
            if class_id:
                students = [s for s in students if s.get('class_id') == class_id]
            
            return True, "Students retrieved successfully", students
            
        except Exception as e:
            logger.error(f"Get students failed: {e}")
            return False, f"Failed to get students: {str(e)}", []
    
    def get_student(self, student_id: str) -> Tuple[bool, str, Dict]:
        """
        Get student by ID
        
        Args:
            student_id: Student ID
            
        Returns:
            (success, message, student_info)
        """
        try:
            student_info = self.faiss_manager.get_student_info(student_id)
            
            if not student_info:
                return False, f"Student {student_id} not found", {}
            
            return True, "Student retrieved successfully", student_info
            
        except Exception as e:
            logger.error(f"Get student failed: {e}")
            return False, f"Failed to get student: {str(e)}", {}
    
    def update_student(
        self, 
        student_id: str, 
        class_id: Optional[str] = None,
        images: Optional[List[bytes]] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Update student information and/or images
        
        Args:
            student_id: Student ID
            class_id: New class ID (optional)
            images: New images (optional)
            
        Returns:
            (success, message, student_info)
        """
        try:
            # Get current student info
            current_info = self.faiss_manager.get_student_info(student_id)
            if not current_info:
                return False, f"Student {student_id} not found", {}
            
            old_class_id = current_info.get('class_id')
            
            # Update class if changed
            if class_id and class_id != old_class_id:
                if not self.class_manager.class_exists(class_id):
                    return False, f"Class {class_id} does not exist", {}
                
                # Update class counts
                self.class_manager.decrement_student_count(old_class_id)
                self.class_manager.increment_student_count(class_id)
            
            # Process new images if provided
            if images:
                # Remove old embeddings
                self.faiss_manager.remove_by_student_id(student_id)
                
                # Process new images
                embeddings = []
                valid_images = []
                
                for i, image_bytes in enumerate(images):
                    try:
                        image = load_image_from_bytes(image_bytes)
                        if not check_image_quality(image):
                            continue
                        
                        faces = self.app.get(image)
                        if not faces:
                            continue
                        
                        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        
                        if is_face_too_small(largest_face.bbox, image.shape):
                            continue
                        
                        embedding = largest_face.embedding
                        embeddings.append(embedding)
                        valid_images.append(image_bytes)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process new image {i+1}: {e}")
                        continue
                
                if not embeddings:
                    return False, "No valid faces found in new images", {}
                
                # Save new images
                student_dir = settings.STUDENT_IMAGES_DIR / student_id
                student_dir.mkdir(parents=True, exist_ok=True)
                
                # Clear old images
                for old_file in student_dir.glob("*.jpg"):
                    old_file.unlink()
                
                # Save new images
                for i, image_bytes in enumerate(valid_images):
                    image_path = student_dir / f"image_{i+1}.jpg"
                    save_image(image_bytes, image_path)
                
                # Add new embeddings
                embeddings_array = np.array(embeddings)
                self.faiss_manager.add_embeddings(
                    embeddings=embeddings_array,
                    student_id=student_id,
                    class_id=class_id or old_class_id
                )
            
            # Update metadata if provided
            if class_id:
                # Update in Faiss metadata
                self.faiss_manager.update_student_metadata(
                    student_id=student_id,
                    class_id=class_id
                )
            
            # Save data
            self.faiss_manager.save_index()
            self.class_manager.save_classes()
            
            # Get updated student info
            updated_info = self.faiss_manager.get_student_info(student_id)
            
            return True, "Student updated successfully", updated_info
            
        except Exception as e:
            logger.error(f"Student update failed: {e}")
            return False, f"Update failed: {str(e)}", {}
    
    def delete_student(self, student_id: str) -> Tuple[bool, str]:
        """
        Delete student and all associated data
        
        Args:
            student_id: Student ID
            
        Returns:
            (success, message)
        """
        try:
            # Get student info
            student_info = self.faiss_manager.get_student_info(student_id)
            if not student_info:
                return False, f"Student {student_id} not found"
            
            class_id = student_info.get('class_id')
            
            # Remove from Faiss index
            self.faiss_manager.remove_by_student_id(student_id)
            
            # Update class count
            if class_id:
                self.class_manager.decrement_student_count(class_id)
            
            # Delete images
            student_dir = settings.STUDENT_IMAGES_DIR / student_id
            if student_dir.exists():
                import shutil
                shutil.rmtree(student_dir)
            
            # Save data
            self.faiss_manager.save_index()
            self.class_manager.save_classes()
            
            logger.info(f"Student {student_id} deleted successfully")
            return True, f"Student {student_id} deleted successfully"
            
        except Exception as e:
            logger.error(f"Student deletion failed: {e}")
            return False, f"Deletion failed: {str(e)}"

    def delete_students_by_class(self, class_id: str) -> Tuple[bool, str, int]:
        """
        Delete all students belonging to a specific class
        
        Returns:
            (success, message, num_deleted)
        """
        try:
            students = self.faiss_manager.get_all_students()
            to_delete = [s.get('student_id') for s in students if s.get('class_id') == class_id]
            deleted = 0
            for student_id in to_delete:
                ok, _ = self.delete_student(student_id)
                if ok:
                    deleted += 1
            return True, "Students deleted successfully", deleted
        except Exception as e:
            logger.error(f"Delete students by class failed: {e}")
            return False, f"Delete students by class failed: {str(e)}", 0
    
    def recognize_faces(
        self, 
        image_bytes: bytes, 
        class_filter: Optional[str] = None,
        top_k: int = 5
    ) -> Tuple[bool, str, List[Dict], int]:
        """
        Recognize faces in an image
        
        Args:
            image_bytes: Image bytes
            class_filter: Optional class ID filter
            top_k: Number of top matches to return
            
        Returns:
            (success, message, matched_students, unmatched_faces)
        """
        try:
            start_time = time.time()
            
            # Load and process image
            image = load_image_from_bytes(image_bytes)
            faces = self.app.get(image)
            
            if not faces:
                return True, "No faces detected", [], 0
            
            matched_students = []
            unmatched_faces = 0
            
            for face in faces:
                try:
                    # Check face size
                    if is_face_too_small(face.bbox, image.shape):
                        unmatched_faces += 1
                        continue
                    
                    # Extract embedding
                    embedding = face.embedding.reshape(1, -1)
                    
                    # Search in Faiss
                    similarities, indices = self.faiss_manager.search_faces(embedding, top_k)
                    
                    if similarities is None or len(similarities) == 0:
                        unmatched_faces += 1
                        continue
                    
                    # Get best match
                    best_similarity = similarities[0]
                    best_index = indices[0]
                    
                    if best_similarity >= self.recognition_threshold:
                        # Get student info
                        student_info = self.faiss_manager.get_student_info_by_index(best_index)
                        
                        if student_info:
                            # Apply class filter if specified
                            if class_filter and student_info.get('class_id') != class_filter:
                                unmatched_faces += 1
                                continue
                            
                            matched_students.append({
                                'student_id': student_info.get('student_id'),
                                'class_id': student_info.get('class_id'),
                                'similarity': float(best_similarity),
                                'confidence': float(best_similarity),
                                'bbox': face.bbox.tolist()
                            })
                        else:
                            unmatched_faces += 1
                    else:
                        unmatched_faces += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process face: {e}")
                    unmatched_faces += 1
                    continue
            
            processing_time = time.time() - start_time
            message = f"Recognized {len(matched_students)} students out of {len(faces)} faces in {processing_time:.2f}s"
            
            return True, message, matched_students, unmatched_faces
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return False, f"Recognition failed: {str(e)}", [], 0
    
    def get_stats(self) -> Tuple[bool, str, Dict]:
        """
        Get system statistics
        
        Returns:
            (success, message, stats)
        """
        try:
            stats = {
                'model_loaded': self.model_loaded,
                'model_name': self.model_name,
                'detection_threshold': self.detection_threshold,
                'recognition_threshold': self.recognition_threshold,
                'total_students': self.faiss_manager.get_total_students(),
                'total_embeddings': self.faiss_manager.get_total_embeddings(),
                'embedding_dimension': settings.EMBEDDING_DIMENSION,
                'index_type': 'IndexFlatIP'
            }
            
            return True, "Stats retrieved successfully", stats
            
        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            return False, f"Failed to get stats: {str(e)}", {}
    
    def get_system_stats(self) -> Tuple[bool, str, Dict]:
        """Alias for get_stats for backward compatibility"""
        return self.get_stats()
    
    def rebuild_index(self) -> Tuple[bool, str, int]:
        """
        Rebuild Faiss index for better performance
        
        Returns:
            (success, message, total_embeddings)
        """
        try:
            total_embeddings = self.faiss_manager.rebuild_index()
            return True, "Index rebuilt successfully", total_embeddings
            
        except Exception as e:
            logger.error(f"Rebuild index failed: {e}")
            return False, f"Rebuild failed: {str(e)}", 0


# Global controller instance
_controller = None
_controller_lock = threading.Lock()

def get_face_recognition_controller() -> FaceRecognitionController:
    """Get the global face recognition controller instance"""
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = FaceRecognitionController()
    return _controller