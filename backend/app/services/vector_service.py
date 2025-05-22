import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from app.models.models import Document, DocumentChunk
from app.config import settings
import logging
import uuid

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection_name = "document_chunks"
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing ChromaDB collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for semantic search"}
            )
            logger.info(f"Initialized collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collection: {str(e)}")
            raise
    
    async def add_document_chunks(self, db: Session, document_id: int) -> bool:
        """Add document chunks to vector database."""
        try:
            # Get document chunks from database
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return False
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                
                documents.append(chunk.content)
                metadatas.append({
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "page_number": chunk.page_number,
                    "paragraph_number": chunk.paragraph_number,
                    "chunk_index": chunk.chunk_index
                })
                ids.append(chunk_id)
                
                # Update chunk with embedding ID
                chunk.embedding_id = chunk_id
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Commit changes to database
            db.commit()
            
            logger.info(f"Added {len(chunks)} chunks for document {document_id} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document chunks to vector database: {str(e)}")
            db.rollback()
            return False
    
    def search_similar_chunks(self, query: str, n_results: int = 10, 
                            document_ids: Optional[List[int]] = None) -> List[Dict]:
        """Search for similar chunks using semantic search."""
        try:
            # Prepare where clause for filtering by document IDs
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, document in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    formatted_results.append({
                        'content': document,
                        'document_id': metadata['document_id'],
                        'chunk_id': metadata['chunk_id'],
                        'page_number': metadata['page_number'],
                        'paragraph_number': metadata['paragraph_number'],
                        'chunk_index': metadata['chunk_index'],
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'embedding_id': results['ids'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def remove_document_chunks(self, db: Session, document_id: int) -> bool:
        """Remove document chunks from vector database."""
        try:
            # Get embedding IDs for the document
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding_id.isnot(None)
            ).all()
            
            if not chunks:
                return True
            
            # Get embedding IDs
            embedding_ids = [chunk.embedding_id for chunk in chunks]
            
            # Remove from ChromaDB
            self.collection.delete(ids=embedding_ids)
            
            logger.info(f"Removed {len(embedding_ids)} chunks for document {document_id} from vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document chunks from vector database: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector database collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_chunks": 0, "collection_name": self.collection_name}
    
    async def reindex_all_documents(self, db: Session) -> bool:
        """Reindex all processed documents in the vector database."""
        try:
            # Clear existing collection
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            
            # Get all processed documents
            documents = db.query(Document).filter(Document.processed == True).all()
            
            success_count = 0
            for document in documents:
                if await self.add_document_chunks(db, document.id):
                    success_count += 1
            
            logger.info(f"Reindexed {success_count}/{len(documents)} documents")
            return success_count == len(documents)
            
        except Exception as e:
            logger.error(f"Error reindexing documents: {str(e)}")
            return False