from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.apis.models.BaseModel import BaseDocument


# ============================================================================
# Class Models
# ============================================================================

class ClassInfo(BaseDocument):
    """Class information model"""
    class_id: str = Field(..., description="Unique class ID")
    total_students: int = Field(0, description="Total number of students")
    
    class Config:
        json_schema_extra = {
            "example": {
                "class_id": "10A1",
                "total_students": 35
            }
        }


class ClassCreateRequest(BaseModel):
    """Request model for creating a class"""
    class_id: str = Field(..., description="Unique class ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "class_id": "10A1"
            }
        }


class ClassUpdateRequest(BaseModel):
    """Request model for updating a class"""
    pass  # No fields needed since we only use class_id


class ClassResponse(BaseModel):
    """Response model for class operations"""
    success: bool
    message: str
    class_info: Optional[ClassInfo] = None


class ClassListResponse(BaseModel):
    """Response model for listing classes"""
    success: bool
    total: int
    classes: List[ClassInfo]


class ClassDetailResponse(BaseModel):
    """Response model for class details with students"""
    success: bool
    class_info: ClassInfo
    students: List[Dict[str, Any]]


# ============================================================================
# Student Models
# ============================================================================

class StudentInfo(BaseDocument):
    """Student information model"""
    student_id: str = Field(..., description="Unique student ID")
    class_id: str = Field(..., description="Class ID")
    num_images: int = Field(0, description="Number of face images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "SV001",
                "class_id": "10A1",
                "num_images": 3
            }
        }


class StudentRegisterRequest(BaseModel):
    """Request model for student registration"""
    student_id: str = Field(..., description="Unique student ID")
    class_id: str = Field(..., description="Class ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "SV001",
                "class_id": "10A1"
            }
        }


class StudentUpdateRequest(BaseModel):
    """Request model for updating student"""
    class_id: Optional[str] = Field(None, description="Class ID")


class StudentResponse(BaseModel):
    """Response model for student operations"""
    success: bool
    message: str
    student: Optional[StudentInfo] = None


class StudentListResponse(BaseModel):
    """Response model for listing students"""
    success: bool
    total: int
    students: List[StudentInfo]


# ============================================================================
# Attendance Models
# ============================================================================

class AttendanceResponse(BaseModel):
    """Response model for attendance (class_id + image)"""
    success: bool
    message: str
    match_found: bool
    student_id: Optional[str] = None
    class_id: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Student recognized successfully",
                "match_found": True,
                "student_id": "SV001",
                "class_id": "10A1",
                "confidence": 0.87,
                "processing_time": 1.23
            }
        }


class FaceMatchResult(BaseModel):
    """Face match result"""
    student_id: str
    class_id: str
    similarity: float
    confidence: float
    bbox: List[float]


# ============================================================================
# System Models
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    model_loaded: bool
    total_students: int
    embedding_dimension: int


class RebuildIndexResponse(BaseModel):
    """Rebuild index response"""
    success: bool
    message: str
    total_embeddings: int
    processing_time: float