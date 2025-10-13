import faiss
import numpy as np
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from loguru import logger
import threading


class FaissManager:
    """
    Optimized Faiss vector database manager for face embeddings
    Uses IndexFlatIP (Inner Product) for cosine similarity search
    Thread-safe implementation with performance optimizations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        """Initialize Faiss Manager"""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Initialize empty index
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []
        self.student_id_to_indices = {}  # Map student_id to list of indices
        self.index_to_student_id = {}    # Map index to student_id
        
        self._initialized = True
        logger.info(f"FaissManager initialized with embedding dimension {embedding_dim}")
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        student_id: str, 
        class_id: str
    ) -> bool:
        """
        Add embeddings to the index
        
        Args:
            embeddings: Array of embeddings (n_embeddings, embedding_dim)
            student_id: Student ID
            class_id: Class ID
            
        Returns:
            bool: Success status
        """
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Get current index size
            current_size = self.index.ntotal
            
            # Add to index
            self.index.add(embeddings)
            
            # Add metadata for each embedding
            for i, embedding in enumerate(embeddings):
                index_id = current_size + i
                
                metadata_entry = {
                    'student_id': student_id,
                    'class_id': class_id,
                    'index_id': index_id,
                    'created_at': int(time.time())
                }
                
                self.metadata.append(metadata_entry)
                
                # Update mappings
                if student_id not in self.student_id_to_indices:
                    self.student_id_to_indices[student_id] = []
                self.student_id_to_indices[student_id].append(index_id)
                self.index_to_student_id[index_id] = student_id
            
            logger.info(f"Added {len(embeddings)} embeddings for student {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def search_faces(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Search for similar faces
        
        Args:
            query_embedding: Query embedding (1, embedding_dim)
            top_k: Number of top results to return
            
        Returns:
            (similarities, indices) or (None, None) if no results
        """
        try:
            if self.index.ntotal == 0:
                return None, None
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            return similarities[0], indices[0]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None, None
    
    def remove_by_student_id(self, student_id: str) -> bool:
        """
        Remove all embeddings for a student
        
        Args:
            student_id: Student ID to remove
            
        Returns:
            bool: Success status
        """
        try:
            if student_id not in self.student_id_to_indices:
                logger.warning(f"Student {student_id} not found in index")
                return True
            
            # Get indices to remove
            indices_to_remove = self.student_id_to_indices[student_id]
            
            if not indices_to_remove:
                return True
            
            # Clean up mappings BEFORE rebuild to avoid KeyError after dict replacement
            if student_id in self.student_id_to_indices:
                del self.student_id_to_indices[student_id]
            for idx in indices_to_remove:
                if idx in self.index_to_student_id:
                    del self.index_to_student_id[idx]

            # Rebuild index without the removed embeddings
            self._rebuild_index_exclude(indices_to_remove)
            
            logger.info(f"Removed {len(indices_to_remove)} embeddings for student {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove student {student_id}: {e}")
            return False
    
    def _rebuild_index_exclude(self, indices_to_exclude: List[int]):
        """Rebuild index excluding specified indices"""
        try:
            # Get all embeddings except the ones to exclude
            all_embeddings = []
            new_metadata = []
            new_student_id_to_indices = {}
            new_index_to_student_id = {}
            
            for i, metadata in enumerate(self.metadata):
                if metadata['index_id'] not in indices_to_exclude:
                    # Get embedding from current index
                    embedding = self.index.reconstruct(metadata['index_id'])
                    all_embeddings.append(embedding)
                    
                    # Update metadata with new index
                    new_index_id = len(all_embeddings) - 1
                    metadata['index_id'] = new_index_id
                    new_metadata.append(metadata)
                    
                    # Update mappings
                    student_id = metadata['student_id']
                    if student_id not in new_student_id_to_indices:
                        new_student_id_to_indices[student_id] = []
                    new_student_id_to_indices[student_id].append(new_index_id)
                    new_index_to_student_id[new_index_id] = student_id
            
            # Create new index
            if all_embeddings:
                embeddings_array = np.array(all_embeddings)
                faiss.normalize_L2(embeddings_array)
                
                new_index = faiss.IndexFlatIP(self.embedding_dim)
                new_index.add(embeddings_array)
                
                # Replace old index
                self.index = new_index
                self.metadata = new_metadata
                self.student_id_to_indices = new_student_id_to_indices
                self.index_to_student_id = new_index_to_student_id
            else:
                # Empty index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.metadata = []
                self.student_id_to_indices = {}
                self.index_to_student_id = {}
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information by student ID"""
        try:
            if student_id not in self.student_id_to_indices:
                return None
            
            # Get first metadata entry for this student
            for metadata in self.metadata:
                if metadata['student_id'] == student_id:
                    return {
                        'student_id': metadata['student_id'],
                        'class_id': metadata['class_id'],
                        'num_images': len(self.student_id_to_indices[student_id]),
                        'created_at': metadata['created_at']
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get student info for {student_id}: {e}")
            return None
    
    def get_student_info_by_index(self, index_id: int) -> Optional[Dict]:
        """Get student information by index ID"""
        try:
            if index_id >= len(self.metadata):
                return None
            
            metadata = self.metadata[index_id]
            student_id = metadata['student_id']
            
            return {
                'student_id': metadata['student_id'],
                'class_id': metadata['class_id'],
                'num_images': len(self.student_id_to_indices.get(student_id, [])),
                'created_at': metadata['created_at']
            }
            
        except Exception as e:
            logger.error(f"Failed to get student info by index {index_id}: {e}")
            return None
    
    def get_all_students(self) -> List[Dict]:
        """Get all students"""
        try:
            students = []
            processed_students = set()
            
            for metadata in self.metadata:
                student_id = metadata['student_id']
                if student_id not in processed_students:
                    students.append({
                        'student_id': metadata['student_id'],
                        'class_id': metadata['class_id'],
                        'num_images': len(self.student_id_to_indices.get(student_id, [])),
                        'created_at': metadata['created_at']
                    })
                    processed_students.add(student_id)
            
            return students
            
        except Exception as e:
            logger.error(f"Failed to get all students: {e}")
            return []
    
    def update_student_metadata(
        self, 
        student_id: str, 
        class_id: Optional[str] = None
    ):
        """Update student metadata"""
        try:
            for metadata in self.metadata:
                if metadata['student_id'] == student_id:
                    if class_id is not None:
                        metadata['class_id'] = class_id
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {student_id}: {e}")
    
    def get_total_students(self) -> int:
        """Get total number of students"""
        return len(self.student_id_to_indices)
    
    def get_total_embeddings(self) -> int:
        """Get total number of embeddings"""
        return self.index.ntotal
    
    def save_index(self) -> bool:
        """Save index and metadata to disk"""
        try:
            if self.index_path and self.metadata_path:
                # Save Faiss index
                faiss.write_index(self.index, str(self.index_path))
                
                # Save metadata
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata': self.metadata,
                        'student_id_to_indices': self.student_id_to_indices,
                        'index_to_student_id': self.index_to_student_id
                    }, f)
                
                logger.info("Index and metadata saved successfully")
                return True
            else:
                logger.warning("Index or metadata path not set")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load index and metadata from disk"""
        try:
            if self.index_path and self.index_path.exists():
                # Load Faiss index
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded Faiss index with {self.index.ntotal} embeddings")
            else:
                logger.info("No existing index found, starting with empty index")
                return True
            
            if self.metadata_path and self.metadata_path.exists():
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', [])
                    self.student_id_to_indices = data.get('student_id_to_indices', {})
                    self.index_to_student_id = data.get('index_to_student_id', {})
                
                logger.info(f"Loaded metadata for {len(self.student_id_to_indices)} students")
                return True
            else:
                logger.warning("No metadata file found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def rebuild_index(self) -> int:
        """Rebuild index for better performance"""
        try:
            # Get all current data
            all_embeddings = []
            new_metadata = []
            new_student_id_to_indices = {}
            new_index_to_student_id = {}
            
            for i in range(self.index.ntotal):
                try:
                    embedding = self.index.reconstruct(i)
                    all_embeddings.append(embedding)
                    
                    # Find metadata for this index
                    metadata = None
                    for m in self.metadata:
                        if m['index_id'] == i:
                            metadata = m
                            break
                    
                    if metadata:
                        # Update metadata with new index
                        new_index_id = len(all_embeddings) - 1
                        metadata['index_id'] = new_index_id
                        new_metadata.append(metadata)
                        
                        # Update mappings
                        student_id = metadata['student_id']
                        if student_id not in new_student_id_to_indices:
                            new_student_id_to_indices[student_id] = []
                        new_student_id_to_indices[student_id].append(new_index_id)
                        new_index_to_student_id[new_index_id] = student_id
                        
                except Exception as e:
                    logger.warning(f"Failed to reconstruct embedding {i}: {e}")
                    continue
            
            # Create new optimized index
            if all_embeddings:
                embeddings_array = np.array(all_embeddings)
                faiss.normalize_L2(embeddings_array)
                
                new_index = faiss.IndexFlatIP(self.embedding_dim)
                new_index.add(embeddings_array)
                
                # Replace old index
                self.index = new_index
                self.metadata = new_metadata
                self.student_id_to_indices = new_student_id_to_indices
                self.index_to_student_id = new_index_to_student_id
                
                logger.info(f"Index rebuilt successfully with {len(all_embeddings)} embeddings")
                return len(all_embeddings)
            else:
                logger.warning("No embeddings to rebuild")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return 0


# Global manager instance
_manager = None
_manager_lock = threading.Lock()

def get_faiss_manager() -> FaissManager:
    """Get the global Faiss manager instance"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                from src.config import settings
                _manager = FaissManager(
                    embedding_dim=settings.EMBEDDING_DIMENSION,
                    index_path=settings.EMBEDDINGS_DIR / "faiss_index.bin",
                    metadata_path=settings.EMBEDDINGS_DIR / "metadata.pkl"
                )
    return _manager