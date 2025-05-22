from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    processed = Column(Boolean, default=False)
    page_count = Column(Integer, default=0)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    queries = relationship("QueryResult", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    paragraph_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)  # ChromaDB ID
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    results = relationship("QueryResult", back_populates="query")
    themes = relationship("IdentifiedTheme", back_populates="query")

class QueryResult(Base):
    __tablename__ = "query_results"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    extracted_answer = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    paragraph_number = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Relationships
    query = relationship("Query", back_populates="results")
    document = relationship("Document", back_populates="queries")

class IdentifiedTheme(Base):
    __tablename__ = "identified_themes"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False)
    theme_name = Column(String, nullable=False)
    theme_description = Column(Text, nullable=False)
    supporting_documents = Column(Text, nullable=False)  # JSON string of document IDs
    confidence_score = Column(Float, nullable=False)
    
    # Relationships
    query = relationship("Query", back_populates="themes")