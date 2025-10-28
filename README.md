# Face Recognition API — Student Attendance, Supercharged

Fast, production-ready face recognition backend for student attendance. Built with FastAPI, powered by InsightFace (ArcFace) for embeddings and Faiss for vector search. Includes student and class management, robust persistence, and clean REST APIs.

## Why This Project

- **Accurate**: Uses InsightFace ArcFace models for strong recognition performance
- **Fast**: Faiss IndexFlatIP with normalized embeddings for cosine similarity search
- **Practical**: Endpoints tailored for real-world attendance flows
- **Resilient**: On-start warmup, graceful shutdown with index persistence
- **Simple**: JSON metadata + file-based storage, easy to back up and restore

## Architecture at a Glance

- `FastAPI` app (`src/apis/create_app.py`) mounts routers under `/api`
- `FaceRecognitionController` singleton handles model loading and high-level ops
- `FaissManager` singleton manages vector index and metadata persistence
- `ClassManager` handles class creation/listing/deletion and counts
- Startup pre-initialization loads model, index, classes; shutdown saves state

Data layout (defaults in `src/config/settings.py`):
- `face_data/embeddings/faiss_index.bin` — Faiss index
- `face_data/embeddings/metadata.pkl` — Embedding metadata
- `face_data/classes.json` — Class registry and student counts
- `face_data/student_images/{student_id}/image_*.jpg` — Stored face images

## Quick Start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Run the API server
```bash
python app.py
```

3) Open API docs
- Swagger UI: `http://localhost:9001/`
- Health check: `http://localhost:9001/health`

Environment is configurable via `.env` or environment variables.

## Configuration

Key settings in `src/config/settings.py` (env var → default):
- `HOST` → `0.0.0.0`, `PORT` → `9001`
- `INSIGHTFACE_MODEL` → `buffalo_l` (also: `buffalo_s`, `buffalo_sc`)
- `FACE_DETECTION_THRESHOLD` → `0.5`
- `FACE_RECOGNITION_THRESHOLD` → `0.6`
- Paths: `FACE_DATA_DIR`, `EMBEDDINGS_DIR`, `STUDENT_IMAGES_DIR`, `CLASS_METADATA_FILE`
- Optional: Redis/JWT/logging placeholders for future expansion

Example `.env`:
```bash
HOST=0.0.0.0
PORT=9001
INSIGHTFACE_MODEL=buffalo_l
FACE_DETECTION_THRESHOLD=0.5
FACE_RECOGNITION_THRESHOLD=0.6
```

## Core API Endpoints

Base path: `/api`

### Class Management
- `GET /api/classes` — list classes
- `POST /api/classes` — create class
- `GET /api/classes/{class_id}` — class detail + students
- `GET /api/classes/{class_id}/students` — list students in class
- `GET /api/classes/{class_id}/stats` — simple stats for a class
- `DELETE /api/classes/{class_id}` — delete class and ALL its students

Create a class:
```bash
curl -X POST http://localhost:9001/api/classes \
  -H 'Content-Type: application/json' \
  -d '{"class_id":"10A1"}'
```

### Student Management
- `POST /api/face-recognition/students/register` — register a student with images
- `GET /api/face-recognition/students` — list all students (optional `class_id` filter)
- `GET /api/face-recognition/students/{student_id}` — get a student
- `PUT /api/face-recognition/students/{student_id}` — update class and/or images
- `DELETE /api/face-recognition/students/{student_id}` — delete student + images

Register a student (3–5 clear images recommended):
```bash
curl -X POST http://localhost:9001/api/face-recognition/students/register \
  -F student_id=SV001 \
  -F class_id=10A1 \
  -F images=@/path/to/image1.jpg \
  -F images=@/path/to/image2.jpg \
  -F images=@/path/to/image3.jpg
```

### Attendance (Face Recognition)
- `POST /api/face-recognition/attendance` — recognize face within a class

Example:
```bash
curl -X POST http://localhost:9001/api/face-recognition/attendance \
  -F class_id=10A1 \
  -F image=@/path/to/single_face.jpg
```
Response includes `match_found`, `student_id`, `class_id`, `confidence`, and `processing_time`.

### System
- `GET /api/face-recognition/health` — model/index health snapshot
- `GET /api/face-recognition/stats` — system statistics
- `POST /api/face-recognition/rebuild-index` — rebuild Faiss index

## How It Works

1) Images are validated and parsed; faces detected with InsightFace
2) Largest face per image is used; small/low-quality faces are skipped
3) ArcFace embeddings are extracted and L2-normalized
4) Embeddings are added to Faiss `IndexFlatIP` (cosine similarity)
5) Metadata and student → index mappings persist to disk
6) Recognition searches top-K neighbors; a match requires `similarity >= FACE_RECOGNITION_THRESHOLD`

## Storage & Backup

All data lives under `face_data/` by default:
- Rebuildable vector index + metadata: `face_data/embeddings/`
- Human-inspectable classes: `face_data/classes.json`
- Original student images: `face_data/student_images/{student_id}/`

For backups, copy the full `face_data/` folder or use the included `face_data/backup/` structure as a reference.

## Performance Tips

- Prefer `buffalo_l` for best accuracy; switch to `buffalo_s` for faster startup
- Provide 3–5 high-quality, well-lit images per student
- Keep faces large and centered; avoid obstructions and extreme angles
- Rebuild index (`POST /api/face-recognition/rebuild-index`) after large batch updates

## Local Development

```bash
python app.py
# Auto-reload is enabled by default in code (uvicorn reload=True)
```

Optional Streamlit demo is available (if you choose to use it):
- `streamlit_app.py` and `run_streamlit.sh` (adjust as needed)

## Requirements

- Python 3.8+
- FastAPI, Uvicorn
- InsightFace
- Faiss (install platform-specific wheels if needed)

## Frequently Asked Questions

- What if no faces are detected? — The API returns success with `match_found=false` and zero matches.
- Can I restrict recognition to a class? — Yes, attendance endpoint applies a `class_id` filter for higher precision.
- How are thresholds chosen? — Start with defaults; tune `FACE_RECOGNITION_THRESHOLD` (0.55–0.7 typical) based on your dataset.

## Project Structure (key parts)

```
app.py                         # Entry point (Uvicorn)
src/apis/create_app.py         # FastAPI factory, routes mount, startup/shutdown
src/apis/controllers/...       # FaceRecognitionController
src/apis/routes/...            # REST endpoints for classes and recognition
src/apis/models/...            # Pydantic request/response models
src/utils/faiss_manager.py     # Faiss index and metadata manager
src/utils/class_manager.py     # Class registry and counters
src/config/settings.py         # Paths, thresholds, model settings
face_data/                     # Persisted data (index, metadata, images)
```

## License

This project is provided under the MIT License. See `LICENSE` if present, or include your preferred license file.


