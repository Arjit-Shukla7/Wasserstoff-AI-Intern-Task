import openai
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.models import Document, Query, QueryResult, IdentifiedTheme
from app.services.vector_service import VectorService
from app.config import settings
import json
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not provided. AI features will be limited.")
        else:
            openai.api_key = settings.OPENAI_API_KEY
        
        self.vector_service = VectorService()
        self.model = settings.DEFAULT_MODEL
    
    async def process_query(self, db: Session, query_text: str, 
                          selected_documents: Optional[List[int]] = None) -> Dict:
        """Process a user query against documents and identify themes."""
        try:
            # Create query record
            query = Query(query_text=query_text)
            db.add(query)
            db.commit()
            db.refresh(query)
            
            # Get relevant document chunks using vector search
            similar_chunks = self.vector_service.search_similar_chunks(
                query=query_text,
                n_results=50,
                document_ids=selected_documents
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
            for doc_id, chunks in document_chunks.items():
                document = db.query(Document).filter(Document.id == doc_id).first()
                if document:
                    result = await self._process_document_query(
                        db, query.id, document, chunks, query_text
                    )
                    if result:
                        individual_results.append(result)
            
            # Identify themes across all results
            themes = await self._identify_themes(db, query.id, individual_results)
            
            return {
                "query_id": query.id,
                "individual_results": individual_results,
                "themes": themes
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _process_document_query(self, db: Session, query_id: int, 
                                    document: Document, chunks: List[Dict], 
                                    query_text: str) -> Optional[Dict]:
        """Process query against a single document."""
        try:
            # Combine relevant chunks for context
            context_chunks = sorted(chunks[:5], key=lambda x: x['similarity_score'], reverse=True)
            context = "\n\n".join([chunk['content'] for chunk in context_chunks])
            
            # Generate answer using AI
            answer_data = await self._generate_answer(query_text, context, document.original_filename)
            
            if not answer_data:
                return None
            
            # Find the most relevant chunk for citation
            best_chunk = context_chunks[0]
            
            # Store result in database
            query_result = QueryResult(
                query_id=query_id,
                document_id=document.id,
                extracted_answer=answer_data['answer'],
                page_number=best_chunk['page_number'],
                paragraph_number=best_chunk['paragraph_number'],
                confidence_score=answer_data['confidence']
            )
            db.add(query_result)
            db.commit()
            
            return {
                "document_id": document.id,
                "document_filename": document.original_filename,
                "extracted_answer": answer_data['answer'],
                "page_number": best_chunk['page_number'],
                "paragraph_number": best_chunk['paragraph_number'],
                "confidence_score": answer_data['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error processing document query for {document.id}: {str(e)}")
            return None
    
    async def _generate_answer(self, query: str, context: str, document_name: str) -> Optional[Dict]:
        """Generate answer using OpenAI API."""
        try:
            if not settings.OPENAI_API_KEY:
                # Fallback to simple keyword matching if no API key
                return self._fallback_answer_generation(query, context)
            
            prompt = f"""
            You are analyzing a document titled "{document_name}" to answer a specific query.
            
            Query: {query}
            
            Context from document:
            {context}
            
            Please provide:
            1. A direct answer to the query based on the context (if possible)
            2. A confidence score between 0.0 and 1.0
            
            Respond in JSON format:
            {{
                "answer": "Your answer here",
                "confidence": 0.8,
                "relevant": true
            }}
            
            If the context doesn't contain relevant information, set "relevant" to false and provide a low confidence score.
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful document analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get('relevant', False):
                return {
                    "answer": result['answer'],
                    "confidence": result['confidence']
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {str(e)}")
            return self._fallback_answer_generation(query, context)
    
    def _fallback_answer_generation(self, query: str, context: str) -> Dict:
        """Fallback answer generation using simple keyword matching."""
        query_words = set(query.lower().split())
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
        
        return {
            "answer": best_sentence if best_sentence else "No relevant information found.",
            "confidence": confidence
        }
    
    async def _identify_themes(self, db: Session, query_id: int, 
                             individual_results: List[Dict]) -> List[Dict]:
        """Identify common themes across document results."""
        try:
            if not individual_results:
                return []
            
            # Combine all answers for theme analysis
            all_answers = [result['extracted_answer'] for result in individual_results]
            combined_text = "\n\n".join(all_answers)
            
            if not settings.OPENAI_API_KEY:
                return self._fallback_theme_identification(individual_results)
            
            prompt = f"""
            Analyze the following responses from multiple documents to identify common themes.
            
            Document Responses:
            {combined_text}
            
            Identify up to {settings.MAX_THEMES} distinct themes that appear across multiple documents.
            For each theme, provide:
            1. Theme name (concise)
            2. Theme description (detailed explanation)
            3. Supporting evidence from the responses
            4. Confidence score (0.0 to 1.0)
            
            Respond in JSON format:
            {{
                "themes": [
                    {{
                        "name": "Theme Name",
                        "description": "Detailed description",
                        "confidence": 0.8,
                        "supporting_documents": ["DOC001", "DOC002"]
                    }}
                ]
            }}
            
            Only include themes with confidence > {settings.MIN_THEME_CONFIDENCE}.
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying themes and patterns in text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            themes = result.get('themes', [])
            
            # Store themes in database and return formatted results
            formatted_themes = []
            for theme_data in themes:
                if theme_data['confidence'] >= settings.MIN_THEME_CONFIDENCE:
                    # Create document ID mapping
                    supporting_docs = [f"DOC{str(res['document_id']).zfill(3)}" 
                                     for res in individual_results]
                    
                    theme = IdentifiedTheme(
                        query_id=query_id,
                        theme_name=theme_data['name'],
                        theme_description=theme_data['description'],
                        supporting_documents=json.dumps(supporting_docs),
                        confidence_score=theme_data['confidence']
                    )
                    db.add(theme)
                    
                    formatted_themes.append({
                        "theme_name": theme_data['name'],
                        "theme_description": theme_data['description'],
                        "supporting_documents": supporting_docs,
                        "confidence_score": theme_data['confidence']
                    })
            
            db.commit()
            return formatted_themes
            
        except Exception as e:
            logger.error(f"Error identifying themes: {str(e)}")
            return self._fallback_theme_identification(individual_results)
    
    def _fallback_theme_identification(self, individual_results: List[Dict]) -> List[Dict]:
        """Fallback theme identification using keyword clustering."""
        try:
            # Simple keyword-based theme identification
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
                        "confidence_score": min(freq / len(individual_results), 1.0)
                    })
                    processed_words.add(word)
                
                if len(themes) >= settings.MAX_THEMES:
                    break
            
            return themes
            
        except Exception as e:
            logger.error(f"Error in fallback theme identification: {str(e)}")
            return []