import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from loguru import logger
import threading
import time

from src.config import settings


class ClassManager:
    """
    Optimized Class Manager for handling class metadata
    Simplified to only handle essential class operations
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
    
    def __init__(self, classes_file: Optional[Path] = None):
        """Initialize Class Manager"""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.classes_file = classes_file or settings.CLASS_METADATA_FILE
        self.classes = {}  # class_id -> class_info
        self._initialized = True
        logger.info("ClassManager initialized")
    
    def create_class(self, class_id: str) -> bool:
        """
        Create a new class
        
        Args:
            class_id: Unique class ID
            
        Returns:
            bool: Success status
        """
        try:
            if class_id in self.classes:
                logger.warning(f"Class {class_id} already exists")
                return False
            
            class_info = {
                'class_id': class_id,
                'total_students': 0,
                'created_at': int(time.time()),
                'updated_at': int(time.time())
            }
            
            self.classes[class_id] = class_info
            self.save_classes()
            
            logger.info(f"Class {class_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create class {class_id}: {e}")
            return False
    
    def get_class(self, class_id: str) -> Optional[Dict]:
        """Get class information"""
        return self.classes.get(class_id)
    
    def get_all_classes(self) -> List[Dict]:
        """Get all classes"""
        return list(self.classes.values())
    
    def update_class(self, class_id: str) -> bool:
        """
        Update class information (currently no fields to update)
        
        Args:
            class_id: Class ID
            
        Returns:
            bool: Success status
        """
        try:
            if class_id not in self.classes:
                logger.warning(f"Class {class_id} not found")
                return False
            
            self.classes[class_id]['updated_at'] = int(time.time())
            self.save_classes()
            
            logger.info(f"Class {class_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update class {class_id}: {e}")
            return False
    
    def delete_class(self, class_id: str) -> bool:
        """
        Delete class
        
        Args:
            class_id: Class ID
            
        Returns:
            bool: Success status
        """
        try:
            if class_id not in self.classes:
                logger.warning(f"Class {class_id} not found")
                return False
            
            del self.classes[class_id]
            self.save_classes()
            
            logger.info(f"Class {class_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete class {class_id}: {e}")
            return False
    
    def increment_student_count(self, class_id: str) -> bool:
        """Increment student count for a class"""
        try:
            if class_id in self.classes:
                self.classes[class_id]['total_students'] += 1
                self.classes[class_id]['updated_at'] = int(time.time())
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to increment student count for {class_id}: {e}")
            return False
    
    def decrement_student_count(self, class_id: str) -> bool:
        """Decrement student count for a class"""
        try:
            if class_id in self.classes:
                self.classes[class_id]['total_students'] = max(0, self.classes[class_id]['total_students'] - 1)
                self.classes[class_id]['updated_at'] = int(time.time())
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to decrement student count for {class_id}: {e}")
            return False
    
    def class_exists(self, class_id: str) -> bool:
        """Check if class exists"""
        return class_id in self.classes
    
    def save_classes(self) -> bool:
        """Save classes to file"""
        try:
            self.classes_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.classes_file, 'w', encoding='utf-8') as f:
                json.dump(self.classes, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save classes: {e}")
            return False
    
    def load_classes(self) -> bool:
        """Load classes from file"""
        try:
            if not self.classes_file.exists():
                logger.info("No classes file found, starting with empty classes")
                return True
            
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                self.classes = json.load(f)
            
            logger.info(f"Loaded {len(self.classes)} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load classes: {e}")
            return False


# Global manager instance
_manager = None
_manager_lock = threading.Lock()

def get_class_manager() -> ClassManager:
    """Get the global class manager instance"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ClassManager()
    return _manager