import os
import uuid
import pytesseract
import cv2
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from app.models.models import Document, DocumentChunk
from app.models.schemas import DocumentCreate
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    
    async def save_uploaded_file(self, file, filename: str) -> str:
        """Save uploaded file to disk and return file path."""
        file_extension = os.path.splitext(filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, unique_filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return file_path
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF file with page and paragraph information."""
        try:
            reader = PdfReader(file_path)
            pages_content = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                paragraphs = self._split_into_paragraphs(text)
                
                page_content = {
                    'page_number': page_num + 1,
                    'paragraphs': paragraphs
                }
                pages_content.append(page_content)
            
            return pages_content
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise
    
    def extract_text_from_image(self, file_path: str) -> List[Dict]:
        """Extract text from image using OCR."""
        try:
            # Load and preprocess image
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR results
            gray = self._preprocess_image(gray)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(gray)
            paragraphs = self._split_into_paragraphs(text)
            
            return [{
                'page_number': 1,
                'paragraphs': paragraphs
            }]
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Noise removal
        denoised = cv2.medianBlur(image, 5)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _split_into_paragraphs(self, text: str) -> List[Dict]:
        """Split text into paragraphs and return with metadata."""
        paragraphs = []
        lines = text.split('\n')
        current_paragraph = []
        paragraph_number = 1
        
        for line in lines:
            line = line.strip()
            if line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    if len(paragraph_text.strip()) > 0:
                        paragraphs.append({
                            'paragraph_number': paragraph_number,
                            'content': paragraph_text
                        })
                        paragraph_number += 1
                    current_paragraph = []
        
        # Don't forget the last paragraph
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            if len(paragraph_text.strip()) > 0:
                paragraphs.append({
                    'paragraph_number': paragraph_number,
                    'content': paragraph_text
                })
        
        return paragraphs
    
    async def process_document(self, db: Session, document_id: int) -> bool:
        """Process a document and store chunks in database."""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            # Extract text based on file type
            if document.file_type.lower() == 'pdf':
                pages_content = self.extract_text_from_pdf(document.file_path)
            elif document.file_type.lower() in ['png', 'jpg', 'jpeg']:
                pages_content = self.extract_text_from_image(document.file_path)
            else:
                logger.error(f"Unsupported file type: {document.file_type}")
                return False
            
            # Store chunks in database
            chunk_index = 0
            for page_content in pages_content:
                for paragraph in page_content['paragraphs']:
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        page_number=page_content['page_number'],
                        paragraph_number=paragraph['paragraph_number'],
                        content=paragraph['content']
                    )
                    db.add(chunk)
                    chunk_index += 1
            
            # Update document status
            document.processed = True
            document.page_count = len(pages_content)
            db.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            db.rollback()
            return False

class DocumentService:
    def __init__(self):
        self.processor = DocumentProcessor()
    
    async def upload_document(self, db: Session, file, filename: str) -> Document:
        """Upload and save document information to database."""
        try:
            # Save file to disk
            file_path = await self.processor.save_uploaded_file(file, filename)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_extension = os.path.splitext(filename)[1].lower().replace('.', '')
            
            # Create document record
            document_data = DocumentCreate(
                filename=os.path.basename(file_path),
                original_filename=filename,
                file_path=file_path,
                file_type=file_extension,
                file_size=file_size
            )
            
            document = Document(**document_data.model_dump())
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Process document asynchronously
            await self.processor.process_document(db, document.id)
            
            return document
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise
    
    def get_documents(self, db: Session) -> List[Document]:
        """Get all documents from database."""
        return db.query(Document).all()
    
    def get_document_by_id(self, db: Session, document_id: int) -> Document:
        """Get document by ID."""
        return db.query(Document).filter(Document.id == document_id).first()
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """Delete document and its associated files."""
        try:
            document = self.get_document_by_id(db, document_id)
            if not document:
                return False
            
            # Delete file from disk
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # Delete from database (chunks will be deleted due to cascade)
            db.delete(document)
            db.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            db.rollback()
            return False