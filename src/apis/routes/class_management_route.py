from fastapi import APIRouter, HTTPException
from loguru import logger

from src.apis.models.face_models import (
    ClassCreateRequest,
    ClassUpdateRequest,
    ClassResponse,
    ClassListResponse,
    ClassDetailResponse,
    ClassInfo
)
from src.apis.controllers.face_recognition_controller import get_face_recognition_controller
from src.utils.class_manager import get_class_manager

router = APIRouter(prefix="/classes", tags=["Class Management"])


# ============================================================================
# Class Management Endpoints
# ============================================================================

@router.post("", response_model=ClassResponse)
async def create_class(request: ClassCreateRequest):
    """
    Create a new class
    
    - **class_id**: Unique class ID (user-defined, e.g., "10A1")
    """
    try:
        class_manager = get_class_manager()
        
        success = class_manager.create_class(class_id=request.class_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create class")
        
        # Get the created class info
        class_info = class_manager.get_class(request.class_id)
        
        return ClassResponse(
            success=True,
            message="Class created successfully",
            class_info=ClassInfo(**class_info)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create class: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create class: {str(e)}")


@router.get("", response_model=ClassListResponse)
async def get_all_classes():
    """
    Get list of all classes with student counts
    
    - **grade**: Optional filter by grade level
    """
    try:
        class_manager = get_class_manager()
        classes = class_manager.get_all_classes()
        
        # Convert to ClassInfo models
        class_infos = [ClassInfo(**c) for c in classes]
        
        return ClassListResponse(
            success=True,
            total=len(class_infos),
            classes=class_infos
        )
        
    except Exception as e:
        logger.error(f"Failed to get classes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get classes: {str(e)}")


@router.get("/{class_id}", response_model=ClassDetailResponse)
async def get_class_detail(class_id: str):
    """
    Get detailed information about a class including student list
    
    - **class_id**: Class ID to retrieve
    
    Returns class information and list of all students in the class
    """
    try:
        class_manager = get_class_manager()
        controller = get_face_recognition_controller()
        
        # Get class info
        class_info = class_manager.get_class(class_id)
        
        if not class_info:
            raise HTTPException(status_code=404, detail=f"Class {class_id} not found")
        
        # Get students in this class
        students = controller.get_students_by_class(class_id)
        
        # Format student data (only essential fields)
        student_list = [
            {
                "student_id": s.get("student_id"),
                "name": s.get("name"),
                "num_images": s.get("num_images", 0)
            }
            for s in students
        ]
        
        # Update actual student count
        actual_count = len(student_list)
        if class_info.get("total_students") != actual_count:
            class_manager.update_student_count(class_id, actual_count)
            class_info["total_students"] = actual_count
        
        return ClassDetailResponse(
            success=True,
            class_info=ClassInfo(**class_info),
            students=student_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get class detail: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get class detail: {str(e)}")


@router.get("/{class_id}/students")
async def get_class_students(class_id: str):
    """
    Get list of students in a specific class
    
    - **class_id**: Class ID
    
    Returns list of students with their information
    """
    try:
        class_manager = get_class_manager()
        controller = get_face_recognition_controller()
        
        # Check if class exists
        if not class_manager.class_exists(class_id):
            raise HTTPException(status_code=404, detail=f"Class {class_id} not found")
        
        # Get students
        students = controller.get_students_by_class(class_id)
        
        return {
            "success": True,
            "class_id": class_id,
            "total": len(students),
            "students": students
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get class students: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get class students: {str(e)}")


## Update class endpoint removed per requirements: only create/list/delete are supported


@router.delete("/{class_id}")
async def delete_class(class_id: str):
    """
    Delete a class and ALL students in that class
    
    ⚠️ WARNING: This will permanently delete:
    - Class information
    - All students registered in this class
    - All face images of students in this class
    
    - **class_id**: Class ID to delete
    """
    try:
        class_manager = get_class_manager()
        controller = get_face_recognition_controller()
        
        # Check if class exists
        if not class_manager.class_exists(class_id):
            raise HTTPException(status_code=404, detail=f"Class {class_id} not found")
        
        # Delete all students in the class first
        success, message, num_deleted = controller.delete_students_by_class(class_id)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete students: {message}")
        
        # Delete the class
        success = class_manager.delete_class(class_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete class")
        
        return {
            "success": True,
            "message": f"Class {class_id} and {num_deleted} students deleted successfully",
            "students_deleted": num_deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete class: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete class: {str(e)}")


@router.get("/{class_id}/stats")
async def get_class_stats(class_id: str):
    """
    Get statistics for a specific class
    
    - **class_id**: Class ID
    
    Returns statistics including student count, attendance rate, etc.
    """
    try:
        class_manager = get_class_manager()
        controller = get_face_recognition_controller()
        
        # Get class info
        class_info = class_manager.get_class(class_id)
        
        if not class_info:
            raise HTTPException(status_code=404, detail=f"Class {class_id} not found")
        
        # Get students
        students = controller.get_students_by_class(class_id)
        
        # Calculate stats
        total_students = len(students)
        total_images = sum(s.get("num_images", 0) for s in students)
        avg_images_per_student = total_images / total_students if total_students > 0 else 0
        
        return {
            "success": True,
            "class_id": class_id,
            "total_students": total_students,
            "total_face_images": total_images,
            "avg_images_per_student": round(avg_images_per_student, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get class stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get class stats: {str(e)}")

