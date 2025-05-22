from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str

class DocumentCreate(DocumentBase):
    file_path: str
    file_size: int

class DocumentResponse(DocumentBase):
    id: int
    upload_date: datetime
    processed: bool
    page_count: int
    
    class Config:
        from_attributes = True

class DocumentChunkResponse(BaseModel):
    id: int
    chunk_index: int
    page_number: int
    paragraph_number: int
    content: str
    
    class Config:
        from_attributes = True

class QueryCreate(BaseModel):
    query_text: str = Field(..., min_length=1, max_length=1000)
    selected_documents: Optional[List[int]] = None

class QueryResultResponse(BaseModel):
    document_id: int
    document_filename: str
    extracted_answer: str
    page_number: int
    paragraph_number: int
    confidence_score: float
    
    class Config:
        from_attributes = True

class ThemeResponse(BaseModel):
    theme_name: str
    theme_description: str
    supporting_documents: List[str]
    confidence_score: float
    
    class Config:
        from_attributes = True

class QueryResponse(BaseModel):
    id: int
    query_text: str
    timestamp: datetime
    individual_results: List[QueryResultResponse]
    identified_themes: List[ThemeResponse]
    
    class Config:
        from_attributes = True

class FileUploadResponse(BaseModel):
    filename: str
    file_size: int
    status: str
    message: str

class ProcessingStatus(BaseModel):
    total_documents: int
    processed_documents: int
    status: str
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str