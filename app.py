from dotenv import load_dotenv

load_dotenv()

from src.apis.create_app import create_app
import uvicorn
from src.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

