import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./chatbot.db"
    
    # OpenAI API
    OPENAI_API_KEY: Optional[str] = None
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # File Upload
    UPLOAD_DIRECTORY: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}
    
    # OCR
    TESSERACT_CMD: Optional[str] = None  # Set if tesseract is not in PATH
    
    # Application
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Models
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Theme identification
    MIN_THEME_CONFIDENCE: float = 0.3
    MAX_THEMES: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)