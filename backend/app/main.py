from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastAPI.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from app.core.database import init_db
from app.api.documents import router as documents_router
from app.api.query import router as query_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()
    logger.info("Application startup complete")
    yield
    # Shutdown
    logger.info("Application shutdown")

# Create FastAPI app
app = FastAPI(
    title="Document Research & Theme Identification Chatbot",
    description="AI-powered chatbot for document analysis and theme identification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(query_router, prefix="/api/query", tags=["query"])

# Mount static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {
        "message": "Document Research & Theme Identification Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "documents": "/api/documents",
            "query": "/api/query",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, 
        log_level="info"
    )