from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from src.apis.routes.face_recognition_route import router as router_face_recognition
from src.apis.routes.class_management_route import router as router_class_management
from loguru import logger

api_router = APIRouter(prefix="/api")
api_router.include_router(router_face_recognition)
api_router.include_router(router_class_management)


def create_app():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Face Recognition API",
        description="Face Recognition System for Student Attendance using InsightFace and Faiss",
        version="1.0.0",
        docs_url="/",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router)

    @app.get("/health")
    async def health_check():
        """Health check endpoint for Docker"""
        return {
            "status": "healthy",
            "message": "Face Recognition API is running",
            "service": "face-recognition"
        }

    @app.on_event("startup")
    async def startup_event():
        """Pre-initialize face recognition controller on server startup"""
        try:
            logger.info("Pre-initializing Face Recognition Controller...")
            from src.apis.controllers.face_recognition_controller import get_face_recognition_controller
            
            # Pre-initialize controller (loads InsightFace model and Faiss index)
            controller = get_face_recognition_controller()
            
            # Log statistics
            success, message, stats = controller.get_stats()
            if success:
                logger.info(f"Face Recognition Controller initialized successfully!")
                logger.info(f"Model: {stats['model_name']}")
                logger.info(f"Total students: {stats['total_students']}")
                logger.info(f"Total embeddings: {stats['total_embeddings']}")
                logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
            else:
                logger.warning(f"Failed to get stats: {message}")
            
            # Log class statistics
            try:
                classes = controller.class_manager.get_all_classes()
                logger.info(f"Total classes: {len(classes)}")
            except Exception as e:
                logger.warning(f"Failed to get class stats: {e}")
            
        except Exception as e:
            logger.error(f"Failed to pre-initialize Face Recognition Controller: {e}")
            logger.warning("Server will start but face recognition may not work properly")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Save index on shutdown"""
        try:
            logger.info("Saving Faiss index before shutdown...")
            from src.apis.controllers.face_recognition_controller import get_face_recognition_controller
            
            controller = get_face_recognition_controller()
            controller.faiss_manager.save_index()
            controller.class_manager.save_classes()
            
            logger.info("Faiss index and class data saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index on shutdown: {e}")

    return app

