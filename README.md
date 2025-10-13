# Face Recognition API

Core face recognition system for student attendance using InsightFace and Faiss.

## Features

- **Student Management**: Register students with face images
- **Class Management**: Create and manage classes
- **Face Recognition**: High-accuracy face recognition using InsightFace
- **Vector Search**: Fast similarity search using Faiss
- **REST API**: Clean REST API endpoints

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start API Server
```bash
python app.py
```

### 3. API Documentation
- **Swagger UI**: http://localhost:9001/
- **Health Check**: http://localhost:9001/health

## API Endpoints

### Classes
- `GET /api/classes` - List all classes
- `POST /api/classes` - Create class
- `GET /api/classes/{class_id}` - Get class details
- `PUT /api/classes/{class_id}` - Update class
- `DELETE /api/classes/{class_id}` - Delete class

### Students
- `POST /api/face-recognition/students/register` - Register student
- `GET /api/face-recognition/students` - List students
- `GET /api/face-recognition/students/{student_id}` - Get student
- `PUT /api/face-recognition/students/{student_id}` - Update student
- `DELETE /api/face-recognition/students/{student_id}` - Delete student

### Attendance
- `POST /api/face-recognition/attendance` - Class attendance check

### System
- `GET /api/face-recognition/health` - Health check
- `GET /api/face-recognition/stats` - System statistics
- `POST /api/face-recognition/rebuild-index` - Rebuild index

## Data Models

### Student
- `student_id`: Unique student ID
- `class_name`: Class name
- `num_images`: Number of face images

### Class
- `class_id`: Unique class ID
- `class_name`: Class name
- `total_students`: Number of students

### Attendance Response
- `success`: Boolean
- `match_found`: Boolean
- `student_id`: Student ID (if found)
- `class_name`: Class name (if found)
- `confidence`: Confidence score (0.0-1.0)
- `processing_time`: Processing time in seconds

## Configuration

Edit `src/config/settings.py` to customize:
- Model settings
- Thresholds
- Storage paths
- Server configuration

## Requirements

- Python 3.8+
- InsightFace
- Faiss
- FastAPI
- Uvicorn
# face_management_system
