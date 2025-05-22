from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
import os
from app.core.database import get_db
from app.models.schemas import DocumentResponse, FileUploadResponse, ProcessingStatus
from app.services.document_service import DocumentService
from app.services.vector_service import VectorService
from app.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
document_service = DocumentService()
vector_service = VectorService()

@router.post("/upload", response_model=FileUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a document for processing."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_extension} not supported. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Upload and process document
        document = await document_service.upload_document(db, file, file.filename)
        
        # Add to vector database
        await vector_service.add_document_chunks(db, document.id)
        
        return FileUploadResponse(
            filename=document.original_filename,
            file_size=document.file_size,
            status="success",
            message="Document uploaded and processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@router.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload multiple documents for processing."""
    try:
        results = []
        successful_uploads = 0
        
        for file in files:
            try:
                # Validate each file
                if not file.filename:
                    results.append({
                        "filename": "unknown",
                        "status": "error",
                        "message": "No filename provided"
                    })
                    continue
                
                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension not in settings.ALLOWED_EXTENSIONS:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": f"File type {file_extension} not supported"
                    })
                    continue
                
                # Check file size
                file.file.seek(0, 2)
                file_size = file.file.tell()
                file.file.seek(0)
                
                if file_size > settings.MAX_FILE_SIZE:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "File size exceeds maximum allowed size"
                    })
                    continue
                
                # Upload and process document
                document = await document_service.upload_document(db, file, file.filename)
                
                # Add to vector database
                await vector_service.add_document_chunks(db, document.id)
                
                results.append({
                    "filename": document.original_filename,
                    "status": "success",
                    "message": "Document uploaded and processed successfully"
                })
                successful_uploads += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Error processing file: {str(e)}"
                })
        
        return {
            "total_files": len(files),
            "successful_uploads": successful_uploads,
            "failed_uploads": len(files) - successful_uploads,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error uploading multiple documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading documents: {str(e)}"
        )

@router.get("/", response_model=List[DocumentResponse])
def get_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all uploaded documents."""
    try:
        documents = document_service.get_documents(db)
        return [DocumentResponse.from_orm(doc) for doc in documents[skip:skip+limit]]
        
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific document by ID."""
    try:
        document = document_service.get_document_by_id(db, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return DocumentResponse.from_orm(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving document: {str(e)}"
        )

@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document and its associated data."""
    try:
        # Remove from vector database first
        vector_service.remove_document_chunks(db, document_id)
        
        # Delete document
        success = document_service.delete_document(db, document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

@router.get("/processing/status", response_model=ProcessingStatus)
def get_processing_status(db: Session = Depends(get_db)):
    """Get document processing status."""
    try:
        documents = document_service.get_documents(db)
        total_documents = len(documents)
        processed_documents = len([doc for doc in documents if doc.processed])
        
        status_text = "completed" if processed_documents == total_documents else "processing"
        
        return ProcessingStatus(
            total_documents=total_documents,
            processed_documents=processed_documents,
            status=status_text,
            message=f"{processed_documents}/{total_documents} documents processed"
        )
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting processing status: {str(e)}"
        )

@router.post("/reindex")
async def reindex_documents(db: Session = Depends(get_db)):
    """Reindex all documents in the vector database."""
    try:
        success = await vector_service.reindex_all_documents(db)
        
        if success:
            return {"message": "All documents reindexed successfully"}
        else:
            return {"message": "Some documents failed to reindex", "status": "partial_success"}
            
    except Exception as e:
        logger.error(f"Error reindexing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reindexing documents: {str(e)}"
        )