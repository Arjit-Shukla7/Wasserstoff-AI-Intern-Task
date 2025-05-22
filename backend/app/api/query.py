from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.schemas import QueryCreate, QueryResponse, QueryResultResponse, ThemeResponse
from app.models.models import Query, QueryResult, IdentifiedTheme, Document
from app.services.ai_service import AIService
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
ai_service = AIService()

@router.post("/query", response_model=dict)
async def process_query(
    query_data: QueryCreate,
    db: Session = Depends(get_db)
):
    """Process a user query against documents."""
    try:
        result = await ai_service.process_query(
            db=db,
            query_text=query_data.query_text,
            selected_documents=query_data.selected_documents
        )
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/queries", response_model=List[QueryResponse])
def get_queries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all queries with their results and themes."""
    try:
        queries = db.query(Query).offset(skip).limit(limit).all()
        
        result = []
        for query in queries:
            # Get individual results
            individual_results = []
            for query_result in query.results:
                document = db.query(Document).filter(Document.id == query_result.document_id).first()
                individual_results.append(QueryResultResponse(
                    document_id=query_result.document_id,
                    document_filename=document.original_filename if document else "Unknown",
                    extracted_answer=query_result.extracted_answer,
                    page_number=query_result.page_number,
                    paragraph_number=query_result.paragraph_number,
                    confidence_score=query_result.confidence_score
                ))
            
            # Get themes
            themes = []
            for theme in query.themes:
                supporting_docs = json.loads(theme.supporting_documents) if theme.supporting_documents else []
                themes.append(ThemeResponse(
                    theme_name=theme.theme_name,
                    theme_description=theme.theme_description,
                    supporting_documents=supporting_docs,
                    confidence_score=theme.confidence_score
                ))
            
            result.append(QueryResponse(
                id=query.id,
                query_text=query.query_text,
                timestamp=query.timestamp,
                individual_results=individual_results,
                identified_themes=themes
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving queries: {str(e)}"
        )

@router.get("/queries/{query_id}", response_model=QueryResponse)
def get_query(
    query_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific query with its results and themes."""
    try:
        query = db.query(Query).filter(Query.id == query_id).first()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        # Get individual results
        individual_results = []
        for query_result in query.results:
            document = db.query(Document).filter(Document.id == query_result.document_id).first()
            individual_results.append(QueryResultResponse(
                document_id=query_result.document_id,
                document_filename=document.original_filename if document else "Unknown",
                extracted_answer=query_result.extracted_answer,
                page_number=query_result.page_number,
                paragraph_number=query_result.paragraph_number,
                confidence_score=query_result.confidence_score
            ))
        
        # Get themes
        themes = []
        for theme in query.themes:
            supporting_docs = json.loads(theme.supporting_documents) if theme.supporting_documents else []
            themes.append(ThemeResponse(
                theme_name=theme.theme_name,
                theme_description=theme.theme_description,
                supporting_documents=supporting_docs,
                confidence_score=theme.confidence_score
            ))
        
        return QueryResponse(
            id=query.id,
            query_text=query.query_text,
            timestamp=query.timestamp,
            individual_results=individual_results,
            identified_themes=themes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query {query_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving query: {str(e)}"
        )

@router.delete("/queries/{query_id}")
def delete_query(
    query_id: int,
    db: Session = Depends(get_db)
):
    """Delete a query and its associated results."""
    try:
        query = db.query(Query).filter(Query.id == query_id).first()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        db.delete(query)
        db.commit()
        
        return {"message": "Query deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query {query_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting query: {str(e)}"
        )