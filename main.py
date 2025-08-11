#HackRx 6.0 - LLM-Powered Intelligent Query-Retrieval System
#FastAPI application for document processing and question answering
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.models import QuestionRequest, AnswerResponse, ProcessingResult
from src.document_processor import AdvancedDocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_service import LLMService
from src.query_processor import QueryProcessor
from src.cache_manager import CacheManager
from src.monitoring import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
doc_processor = None
vector_store = None
llm_service = None
query_processor = None
cache_manager = None
monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global doc_processor, vector_store, llm_service, query_processor, cache_manager, monitor
    
    logger.info("🚀 Starting HackRx 6.0 Application...")
    
    # Initialize components
    doc_processor = AdvancedDocumentProcessor()
    vector_store = VectorStoreManager()
    llm_service = LLMService()
    cache_manager = CacheManager()
    monitor = PerformanceMonitor()
    
    # Initialize vector store
    await vector_store.initialize()
    
    # Initialize query processor with all components
    query_processor = QueryProcessor(
        doc_processor=doc_processor,
        vector_store=vector_store,
        llm_service=llm_service,
        cache_manager=cache_manager,
        monitor=monitor
    )
    
    logger.info("✅ Application initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down application...")
    await vector_store.close()
    await cache_manager.close()
    logger.info("✅ Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 - Intelligent Query-Retrieval System",
    description="LLM-Powered document processing and question answering system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HackRx 6.0 - Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "document_processor": "active",
            "vector_store": "active" if vector_store else "inactive",
            "llm_service": "active" if llm_service else "inactive",
            "cache": "active" if cache_manager else "inactive"
        }
    }

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_questions(
    request: QuestionRequest,
    background_tasks: BackgroundTasks
) -> AnswerResponse:
    """
    Main endpoint for processing documents and answering questions
    
    Args:
        request: QuestionRequest containing document URL and questions
        background_tasks: FastAPI background tasks
        
    Returns:
        AnswerResponse with answers to all questions
    """
    start_time = time.time()
    request_id = hashlib.md5(f"{request.documents}{len(request.questions)}{start_time}".encode()).hexdigest()[:8]
    
    logger.info(f"🔍 [{request_id}] Processing request with {len(request.questions)} questions")
    logger.info(f"📄 [{request_id}] Document: {request.documents[:100]}...")
    
    try:
        # Process questions using the query processor
        results = await query_processor.process_questions(
            document_url=request.documents,
            questions=request.questions,
            request_id=request_id
        )
        
        # Extract answers from results
        answers = [result.answer for result in results]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log performance metrics
        logger.info(f"✅ [{request_id}] Completed in {processing_time:.2f}s")
        logger.info(f"📊 [{request_id}] Average confidence: {sum(r.confidence for r in results) / len(results):.2f}")
        
        # Background task for cleanup and optimization
        background_tasks.add_task(
            _post_processing_cleanup,
            request_id,
            processing_time,
            results
        )
        
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ [{request_id}] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "message": str(e),
                "request_id": request_id
            }
        )

async def _post_processing_cleanup(
    request_id: str,
    processing_time: float,
    results: List[ProcessingResult]
):
    """Background task for post-processing cleanup and optimization"""
    try:
        # Update performance metrics
        if monitor:
            await monitor.record_request(
                request_id=request_id,
                processing_time=processing_time,
                question_count=len(results),
                avg_confidence=sum(r.confidence for r in results) / len(results)
            )
        
        # Cache optimization
        if cache_manager:
            await cache_manager.optimize_cache()
            
        logger.info(f"🧹 [{request_id}] Post-processing cleanup completed")
        
    except Exception as e:
        logger.warning(f"⚠️ [{request_id}] Post-processing cleanup failed: {e}")

@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics
