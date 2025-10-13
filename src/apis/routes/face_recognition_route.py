from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
from datetime import datetime
from loguru import logger

from src.apis.models.face_models import (
    StudentRegisterRequest,
    StudentUpdateRequest,
    StudentResponse,
    StudentListResponse,
    AttendanceResponse,
    HealthCheckResponse,
    RebuildIndexResponse,
    FaceMatchResult
)
from src.apis.controllers.face_recognition_controller import get_face_recognition_controller
from src.utils.face_recognition_utils import generate_session_id, validate_image_format

router = APIRouter(prefix="/face-recognition", tags=["Face Recognition"])


# ============================================================================
# Student Management Endpoints
# ============================================================================

@router.post("/students/register", response_model=StudentResponse)
async def register_student(
    student_id: str = Form(..., description="Unique student ID"),
    class_id: str = Form(..., description="Class ID"),
    images: List[UploadFile] = File(..., description="Face images (3-5 images recommended)")
):
    """
    Register a new student with face images
    
    - **student_id**: Unique identifier for the student
    - **class_id**: Class ID
    - **images**: Multiple face images (JPEG/PNG) from different angles
    
    Recommended: 3-5 clear face images from different angles for better recognition
    """
    try:
        start_time = time.time()
        
        # Validate images
        valid_images = []
        for image in images:
            image_bytes = await image.read()
            is_valid, format_msg = validate_image_format(image_bytes)
            if is_valid:
                valid_images.append(image_bytes)
            else:
                logger.warning(f"Invalid image {image.filename}: {format_msg}")
        
        if not valid_images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Register student
        controller = get_face_recognition_controller()
        success, message, student_info = controller.register_student(
            student_id=student_id,
            class_id=class_id,
            images=valid_images
        )
        
        processing_time = time.time() - start_time
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return StudentResponse(
            success=True,
            message=f"Student registered successfully with {student_info.get('num_images', 0)} images",
            student=student_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/students", response_model=StudentListResponse)
async def get_all_students(
    class_id: Optional[str] = Query(None, description="Filter by class ID")
):
    """
    Get all students with optional class filter
    
    - **class_id**: Optional filter to only return students from specific class
    """
    try:
        controller = get_face_recognition_controller()
        success, message, students = controller.get_all_students(class_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return StudentListResponse(
            success=True,
            total=len(students),
            students=students
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get students failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get students: {str(e)}")


@router.get("/students/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str):
    """
    Get student by ID
    
    - **student_id**: Student ID to retrieve
    """
    try:
        controller = get_face_recognition_controller()
        success, message, student_info = controller.get_student(student_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=message)
        
        return StudentResponse(
            success=True,
            message="Student retrieved successfully",
            student=student_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get student: {str(e)}")


@router.put("/students/{student_id}", response_model=StudentResponse)
async def update_student(
    student_id: str,
    class_id: Optional[str] = Form(None, description="Class ID"),
    images: Optional[List[UploadFile]] = File(None, description="New face images")
):
    """
    Update student information and/or face images
    
    - **student_id**: Student ID to update
    - **class_id**: Updated class ID (optional)
    - **images**: New face images (optional)
    """
    try:
        start_time = time.time()
        
        # Validate images if provided
        valid_images = None
        if images:
            valid_images = []
            for image in images:
                image_bytes = await image.read()
                is_valid, format_msg = validate_image_format(image_bytes)
                if is_valid:
                    valid_images.append(image_bytes)
                else:
                    logger.warning(f"Invalid image {image.filename}: {format_msg}")
        
        # Update student
        controller = get_face_recognition_controller()
        success, message, student_info = controller.update_student(
            student_id=student_id,
            class_id=class_id,
            images=valid_images
        )
        
        processing_time = time.time() - start_time
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return StudentResponse(
            success=True,
            message=f"Student updated successfully",
            student=student_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.delete("/students/{student_id}", response_model=StudentResponse)
async def delete_student(student_id: str):
    """
    Delete student and all associated data
    
    - **student_id**: Student ID to delete
    """
    try:
        controller = get_face_recognition_controller()
        success, message = controller.delete_student(student_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=message)
        
        return StudentResponse(
            success=True,
            message=f"Student {student_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


# ============================================================================
# Attendance Endpoints
# ============================================================================

@router.post("/attendance", response_model=AttendanceResponse)
async def class_attendance_check(
    class_id: str = Form(..., description="Class ID to search in"),
    image: UploadFile = File(..., description="Image containing a single student face")
):
    """
    Class attendance check (class_id + image)
    
    Upload an image with one person's face and specify class_id to get:
    - Student ID
    - Student name  
    - Class name
    - Confidence score
    
    This API searches only within the specified class for better accuracy.
    
    - **class_id**: Class ID to search in (e.g., "10A1")
    - **image**: Image file containing a single face (JPEG/PNG)
    """
    try:
        start_time = time.time()
        
        # Read and validate image
        image_bytes = await image.read()
        is_valid, format_msg = validate_image_format(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Image validation failed: {format_msg}")
        
        # Recognize face within specific class
        controller = get_face_recognition_controller()
        success, message, matched_students, unmatched_count = controller.recognize_faces(
            image_bytes,
            class_filter=class_id,
            top_k=1  # Only get best match
        )
        
        processing_time = time.time() - start_time
        
        if not success:
            return AttendanceResponse(
                success=False,
                message=message,
                match_found=False,
                processing_time=processing_time
            )
        
        # Check if any student matched
        if matched_students and len(matched_students) > 0:
            best_match = matched_students[0]
            return AttendanceResponse(
                success=True,
                message="Student recognized successfully",
                match_found=True,
                student_id=best_match.get('student_id'),
                class_id=best_match.get('class_id'),
                confidence=best_match.get('similarity', 0.0),
                processing_time=processing_time
            )
        else:
            return AttendanceResponse(
                success=True,
                message=f"No matching student found in class {class_id}",
                match_found=False,
                processing_time=processing_time
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Class attendance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Class attendance check failed: {str(e)}")


# ============================================================================
# System Endpoints
# ============================================================================

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    """
    try:
        controller = get_face_recognition_controller()
        success, message, stats = controller.get_system_stats()
        
        return HealthCheckResponse(
            status="healthy" if success else "unhealthy",
            message=message,
            model_loaded=stats.get('model_loaded', False),
            total_students=stats.get('total_students', 0),
            embedding_dimension=stats.get('embedding_dimension', 0)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            model_loaded=False,
            total_students=0,
            embedding_dimension=0
        )


@router.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        controller = get_face_recognition_controller()
        success, message, stats = controller.get_system_stats()
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return {
            "success": True,
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/rebuild-index", response_model=RebuildIndexResponse)
async def rebuild_index():
    """
    Rebuild Faiss index for better performance
    """
    try:
        start_time = time.time()
        
        controller = get_face_recognition_controller()
        success, message, total_embeddings = controller.rebuild_index()
        
        processing_time = time.time() - start_time
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return RebuildIndexResponse(
            success=True,
            message=f"Index rebuilt successfully with {total_embeddings} embeddings",
            total_embeddings=total_embeddings,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild index failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild index failed: {str(e)}")