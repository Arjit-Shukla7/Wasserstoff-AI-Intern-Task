from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session
from langchain.chains import RetrievalQA, LLMChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import VectorStoreRetriever
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
import json
import logging
import re
from collections import defaultdict

# Import your existing models
from app.models.models import Document, Query, QueryResult, IdentifiedTheme
from app.config import settings

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class AnswerResponse(BaseModel):
    answer: str = Field(description="Direct answer to the query")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    relevant: bool = Field(description="Whether the context contains relevant information")

class ThemeResponse(BaseModel):
    name: str = Field(description="Concise theme name")
    description: str = Field(description="Detailed theme description")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    supporting_evidence: List[str] = Field(description="Key supporting evidence")

class ThemesOutput(BaseModel):
    themes: List[ThemeResponse] = Field(description="List of identified themes")

class LangChainAIService:
    def __init__(self):
        """Initialize the LangChain-based AI service."""
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not provided. AI features will be limited.")
            self.llm = None
            self.embeddings = None
        else:
            # Initialize LLM and embeddings
            self.llm = ChatOpenAI(
                model_name=settings.DEFAULT_MODEL,
                temperature=0.1,
                openai_api_key=settings.OPENAI_API_KEY
            )
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY
            )
        
        # Initialize vector store (you'll need to populate this)
        self.vector_store = None
        if self.embeddings:
            self._initialize_vector_store()
        
        # Setup output parsers
        self.answer_parser = PydanticOutputParser(pydantic_object=AnswerResponse)
        self.theme_parser = PydanticOutputParser(pydantic_object=ThemesOutput)
        
        # Create chains
        self._setup_chains()
    
    def _initialize_vector_store(self):
        """Initialize the vector store. In practice, you'd load existing data."""
        try:
            # This would typically connect to your existing vector database
            # For now, we'll create an empty Chroma instance
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None
    
    def _setup_chains(self):
        """Setup LangChain chains for document QA and theme identification."""
        if not self.llm:
            return
        
        # Document QA Chain
        self.qa_prompt = ChatPromptTemplate.from_template("""
        You are analyzing a document titled "{document_name}" to answer a specific query.
        
        Query: {query}
        
        Context from document:
        {context}
        
        Please analyze the context and provide a structured response.
        
        {format_instructions}
        
        If the context doesn't contain relevant information, set "relevant" to false and provide a low confidence score.
        """)
        
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=self.qa_prompt,
            output_parser=OutputFixingParser.from_llm(
                parser=self.answer_parser,
                llm=self.llm
            )
        )
        
        # Theme Analysis Chain - Map step
        map_prompt = PromptTemplate.from_template("""
        Analyze the following document response to identify key themes and concepts:
        
        Document: {document_name}
        Response: {response_text}
        
        Extract key themes, concepts, and topics from this response.
        Focus on actionable insights and important patterns.
        
        Key themes and concepts:
        """)
        
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        
        # Theme Analysis Chain - Reduce step
        reduce_prompt = PromptTemplate.from_template("""
        You are analyzing responses from multiple documents to identify common themes.
        
        Individual theme analyses:
        {doc_summaries}
        
        Original query context: {query}
        
        Identify up to {max_themes} distinct themes that appear across multiple documents.
        Only include themes with high confidence and clear supporting evidence.
        
        {format_instructions}
        """)
        
        reduce_chain = LLMChain(
            llm=self.llm, 
            prompt=reduce_prompt,
            output_parser=OutputFixingParser.from_llm(
                parser=self.theme_parser,
                llm=self.llm
            )
        )
        
        # Combine documents chain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )
        
        # Complete MapReduce chain for theme identification
        self.theme_analysis_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            combine_document_chain=combine_documents_chain,
            document_variable_name="response_text"
        )
    
    async def process_query(self, db: Session, query_text: str, 
                          selected_documents: Optional[List[int]] = None) -> Dict:
        """Process a user query against documents and identify themes."""
        try:
            # Create query record
            query = Query(query_text=query_text)
            db.add(query)
            db.commit()
            db.refresh(query)
            
            # Get relevant document chunks
            similar_chunks = await self._get_similar_chunks(
                query_text, selected_documents
            )
            
            if not similar_chunks:
                logger.warning("No relevant chunks found for query")
                return {
                    "query_id": query.id,
                    "individual_results": [],
                    "themes": []
                }
            
            # Group chunks by document
            document_chunks = defaultdict(list)
            for chunk in similar_chunks:
                document_chunks[chunk['document_id']].append(chunk)
            
            # Process each document individually
            individual_results = []
            with get_openai_callback() as cb:
                for doc_id, chunks in document_chunks.items():
                    document = db.query(Document).filter(Document.id == doc_id).first()
                    if document:
                        result = await self._process_document_query(
                            db, query.id, document, chunks, query_text
                        )
                        if result:
                            individual_results.append(result)
                
                # Identify themes across all results
                themes = await self._identify_themes(db, query.id, individual_results, query_text)
                
                logger.info(f"OpenAI API usage: {cb}")
            
            return {
                "query_id": query.id,
                "individual_results": individual_results,
                "themes": themes
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _get_similar_chunks(self, query_text: str, 
                                selected_documents: Optional[List[int]] = None) -> List[Dict]:
        """Get similar chunks using vector search."""
        if not self.vector_store:
            return self._fallback_chunk_retrieval(query_text, selected_documents)
        
        try:
            # Create retriever
            retriever = VectorStoreRetriever(
                vectorstore=self.vector_store,
                search_kwargs={"k": 50}
            )
            
            # If specific documents are selected, filter the retriever
            if selected_documents:
                # This would require custom retriever implementation
                # For now, we'll retrieve and filter
                docs = retriever.get_relevant_documents(query_text)
                filtered_docs = [
                    doc for doc in docs 
                    if doc.metadata.get('document_id') in selected_documents
                ]
                docs = filtered_docs
            else:
                docs = retriever.get_relevant_documents(query_text)
            
            # Convert to your expected format
            chunks = []
            for doc in docs:
                chunks.append({
                    'document_id': doc.metadata.get('document_id'),
                    'content': doc.page_content,
                    'page_number': doc.metadata.get('page_number', 1),
                    'paragraph_number': doc.metadata.get('paragraph_number', 1),
                    'similarity_score': doc.metadata.get('score', 0.5)
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return self._fallback_chunk_retrieval(query_text, selected_documents)
    
    def _fallback_chunk_retrieval(self, query_text: str, 
                                selected_documents: Optional[List[int]]) -> List[Dict]:
        """Fallback chunk retrieval when vector search is unavailable."""
        # Implement your existing vector service logic here
        # This is a placeholder
        return []
    
    async def _process_document_query(self, db: Session, query_id: int, 
                                    document: Document, chunks: List[Dict], 
                                    query_text: str) -> Optional[Dict]:
        """Process query against a single document using LangChain."""
        try:
            if not self.llm:
                return self._fallback_document_processing(
                    db, query_id, document, chunks, query_text
                )
            
            # Combine relevant chunks for context
            context_chunks = sorted(chunks[:5], key=lambda x: x['similarity_score'], reverse=True)
            context = "\n\n".join([chunk['content'] for chunk in context_chunks])
            
            # Generate answer using LangChain
            response = await self.qa_chain.arun(
                query=query_text,
                context=context,
                document_name=document.original_filename,
                format_instructions=self.answer_parser.get_format_instructions()
            )
            
            if not response.relevant:
                return None
            
            # Find the most relevant chunk for citation
            best_chunk = context_chunks[0]
            
            # Store result in database
            query_result = QueryResult(
                query_id=query_id,
                document_id=document.id,
                extracted_answer=response.answer,
                page_number=best_chunk['page_number'],
                paragraph_number=best_chunk['paragraph_number'],
                confidence_score=response.confidence
            )
            db.add(query_result)
            db.commit()
            
            return {
                "document_id": document.id,
                "document_filename": document.original_filename,
                "extracted_answer": response.answer,
                "page_number": best_chunk['page_number'],
                "paragraph_number": best_chunk['paragraph_number'],
                "confidence_score": response.confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing document query for {document.id}: {str(e)}")
            return self._fallback_document_processing(
                db, query_id, document, chunks, query_text
            )
    
    def _fallback_document_processing(self, db: Session, query_id: int, 
                                    document: Document, chunks: List[Dict], 
                                    query_text: str) -> Optional[Dict]:
        """Fallback document processing without LLM."""
        # Your existing fallback logic here
        try:
            context_chunks = sorted(chunks[:5], key=lambda x: x['similarity_score'], reverse=True)
            context = "\n\n".join([chunk['content'] for chunk in context_chunks])
            
            # Simple keyword matching fallback
            query_words = set(query_text.lower().split())
            context_sentences = context.split('.')
            
            best_sentence = ""
            best_score = 0
            
            for sentence in context_sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence.strip()
            
            confidence = min(best_score / len(query_words), 1.0) if query_words else 0.1
            answer = best_sentence if best_sentence else "No relevant information found."
            
            if confidence < 0.3:
                return None
            
            best_chunk = context_chunks[0]
            
            query_result = QueryResult(
                query_id=query_id,
                document_id=document.id,
                extracted_answer=answer,
                page_number=best_chunk['page_number'],
                paragraph_number=best_chunk['paragraph_number'],
                confidence_score=confidence
            )
            db.add(query_result)
            db.commit()
            
            return {
                "document_id": document.id,
                "document_filename": document.original_filename,
                "extracted_answer": answer,
                "page_number": best_chunk['page_number'],
                "paragraph_number": best_chunk['paragraph_number'],
                "confidence_score": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in fallback document processing: {e}")
            return None
    
    async def _identify_themes(self, db: Session, query_id: int, 
                             individual_results: List[Dict], query_text: str) -> List[Dict]:
        """Identify common themes using LangChain MapReduce chain."""
        try:
            if not individual_results:
                return []
            
            if not self.llm:
                return self._fallback_theme_identification(individual_results)
            
            # Convert results to LangChain documents for MapReduce
            documents = []
            for result in individual_results:
                doc = LangChainDocument(
                    page_content=result['extracted_answer'],
                    metadata={
                        'document_name': result['document_filename'],
                        'document_id': result['document_id']
                    }
                )
                documents.append(doc)
            
            # Run MapReduce theme analysis
            theme_response = await self.theme_analysis_chain.arun(
                input_documents=documents,
                query=query_text,
                max_themes=settings.MAX_THEMES,
                format_instructions=self.theme_parser.get_format_instructions()
            )
            
            # Store themes in database and return formatted results
            formatted_themes = []
            for theme_data in theme_response.themes:
                if theme_data.confidence >= settings.MIN_THEME_CONFIDENCE:
                    # Create document ID mapping
                    supporting_docs = [f"DOC{str(res['document_id']).zfill(3)}" 
                                     for res in individual_results]
                    
                    theme = IdentifiedTheme(
                        query_id=query_id,
                        theme_name=theme_data.name,
                        theme_description=theme_data.description,
                        supporting_documents=json.dumps(supporting_docs),
                        confidence_score=theme_data.confidence
                    )
                    db.add(theme)
                    
                    formatted_themes.append({
                        "theme_name": theme_data.name,
                        "theme_description": theme_data.description,
                        "supporting_documents": supporting_docs,
                        "confidence_score": theme_data.confidence,
                        "supporting_evidence": theme_data.supporting_evidence
                    })
            
            db.commit()
            return formatted_themes
            
        except Exception as e:
            logger.error(f"Error identifying themes with LangChain: {str(e)}")
            return self._fallback_theme_identification(individual_results)
    
    def _fallback_theme_identification(self, individual_results: List[Dict]) -> List[Dict]:
        """Fallback theme identification using keyword clustering."""
        try:
            # Your existing fallback logic
            word_freq = defaultdict(int)
            doc_word_map = defaultdict(list)
            
            for i, result in enumerate(individual_results):
                words = re.findall(r'\b\w+\b', result['extracted_answer'].lower())
                for word in words:
                    if len(word) > 4:  # Only consider longer words
                        word_freq[word] += 1
                        doc_word_map[word].append(f"DOC{str(result['document_id']).zfill(3)}")
            
            # Find themes based on frequent words appearing in multiple documents
            themes = []
            processed_words = set()
            
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
                if word in processed_words or freq < 2:
                    continue
                
                supporting_docs = list(set(doc_word_map[word]))
                if len(supporting_docs) > 1:  # Theme must appear in multiple documents
                    themes.append({
                        "theme_name": word.capitalize() + " Related",
                        "theme_description": f"Theme related to '{word}' appearing across multiple documents",
                        "supporting_documents": supporting_docs,
                        "confidence_score": min(freq / len(individual_results), 1.0),
                        "supporting_evidence": [word]
                    })
                    processed_words.add(word)
                
                if len(themes) >= settings.MAX_THEMES:
                    break
            
            return themes
            
        except Exception as e:
            logger.error(f"Error in fallback theme identification: {str(e)}")
            return []
    
    def add_documents_to_vector_store(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not self.vector_store or not self.embeddings:
            logger.warning("Vector store not available")
            return
        
        try:
            # Convert your documents to LangChain format
            langchain_docs = []
            for doc in documents:
                # You'd need to implement document chunking here
                # This is a simplified example
                content = doc.content  # Assuming you have document content
                
                # Split document into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    langchain_doc = LangChainDocument(
                        page_content=chunk,
                        metadata={
                            'document_id': doc.id,
                            'filename': doc.original_filename,
                            'chunk_index': i,
                            'page_number': 1,  # You'd calculate this
                            'paragraph_number': i + 1
                        }
                    )
                    langchain_docs.append(langchain_doc)
            
            # Add to vector store
            self.vector_store.add_documents(langchain_docs)
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")

# Usage example
"""
# Initialize the service
ai_service = LangChainAIService()

# Process a query
result = await ai_service.process_query(
    db=db_session,
    query_text="What are the main findings?",
    selected_documents=[1, 2, 3]
)
"""