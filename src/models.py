# ============================================================================

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime

class QuestionRequest(BaseModel):
    """Request model for processing questions"""
    documents: str = Field(..., description="PDF document URL to process")
    questions: List[str] = Field(..., description="List of questions to answer")
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        if len(v) > 20:
            raise ValueError('Maximum 20 questions allowed')
        return v
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Document URL must be a valid HTTP/HTTPS URL')
        return v

class AnswerResponse(BaseModel):
    """Response model with answers"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

class ProcessingResult(BaseModel):
    """Internal processing result model"""
    answer: str
    confidence: float
    model: str
    processing_time: float
    validation: Dict[str, Any]
    relevant_chunks: List[str] = []
    metadata: Dict[str, Any] = {}

class DocumentChunk(BaseModel):
    """Document chunk model for vector storage"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
class QueryContext(BaseModel):
    """Context information for query processing"""
    query: str
    document_url: str
    chunk_count: int
    confidence_threshold: float = 0.7
    max_chunks: int = 5

# ============================================================================
