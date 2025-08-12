# ============================================================================

import asyncio
import logging
import time
from typing import List, Dict, Any
from src.models import ProcessingResult

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Main query processing orchestrator"""
    
    def __init__(self, doc_processor, vector_store, llm_service, cache_manager, monitor):
        self.doc_processor = doc_processor
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.cache_manager = cache_manager
        self.monitor = monitor
        
        logger.info("✅ Query processor initialized")
    
    async def process_questions(
        self, 
        document_url: str, 
        questions: List[str],
        request_id: str
    ) -> List[ProcessingResult]:
        """Process all questions for a document"""
        start_time = time.time()
        
        logger.info(f"🔄 [{request_id}] Processing {len(questions)} questions")
        
        # Step 1: Process document (with caching)
        doc_cache_key = self.cache_manager.generate_document_cache_key(document_url)
        cached_doc = await self.cache_manager.get_cached_result(doc_cache_key)
        
        if cached_doc:
            logger.info(f"📋 [{request_id}] Using cached document processing")
            document_processed = True
        else:
            logger.info(f"📄 [{request_id}] Processing document...")
            doc_result = await self.doc_processor.process_document(document_url)
            
            if not doc_result.get('chunks'):
                logger.error(f"❌ [{request_id}] Document processing failed")
                return [self._create_error_result(q, "Document processing failed") for q in questions]
            
            # Store in vector database
            await self.vector_store.store_document_chunks(document_url, doc_result['chunks'])
            
            # Cache the processing result
            await self.cache_manager.cache_result(doc_cache_key, doc_result, ttl=7200)
            document_processed = True
        
        if not document_processed:
            return [self._create_error_result(q, "Document processing failed") for q in questions]
        
        # Step 2: Process questions concurrently
        question_tasks = []
        for question in questions:
            task = self._process_single_question(document_url, question, request_id)
            question_tasks.append(task)
        
        # Execute questions in parallel (with some concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[bounded_task(task) for task in question_tasks],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ [{request_id}] Question {i+1} failed: {result}")
                processed_results.append(self._create_error_result(questions[i], str(result)))
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        logger.info(f"✅ [{request_id}] All questions processed in {total_time:.2f}s")
        
        return processed_results
    
    async def _process_single_question(
        self, 
        document_url: str, 
        question: str,
        request_id: str
    ) -> ProcessingResult:
        """Process a single question"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self.cache_manager.generate_cache_key(document_url, question)
            cached_result = await self.cache_manager.get_cached_result(cache_key)
            
            if cached_result:
                logger.info(f"📋 [{request_id}] Using cached answer for question")
                return ProcessingResult(**cached_result)
            
            # Step 1: Find relevant chunks
            relevant_chunks = await self.vector_store.search_relevant_chunks(
                query=question,
                document_url=document_url,
                top_k=5,
                confidence_threshold=0.3  # Lower threshold for better recall
            )
            
            if not relevant_chunks:
                logger.warning(f"⚠️ [{request_id}] No relevant chunks found for question")
                relevant_chunks = []
            
            # Step 2: Generate answer using LLM
            llm_result = await self.llm_service.answer_question(
                question=question,
                context_chunks=relevant_chunks,
                document_url=document_url
            )
            
            # Step 3: Create result
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                answer=llm_result["answer"],
                confidence=llm_result["confidence"],
                model=llm_result.get("model", "unknown"),
                processing_time=processing_time,
                validation={
                    "chunks_found": len(relevant_chunks),
                    "avg_chunk_similarity": sum(c.get('similarity', 0) for c in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0,
                    "token_usage": llm_result.get("token_usage", {}),
                    "cache_hit": False
                },
                relevant_chunks=[chunk["content"][:200] + "..." for chunk in relevant_chunks[:3]],
                metadata={
                    "question_length": len(question),
                    "answer_length": len(llm_result["answer"]),
                    "processing_steps": ["document_retrieval", "chunk_search", "llm_generation"]
                }
            )
            
            # Cache the result
            await self.cache_manager.cache_result(
                cache_key, 
                result.dict(), 
                ttl=3600  # Cache for 1 hour
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Question processing failed: {e}")
            return self._create_error_result(question, str(e))
    
    def _create_error_result(self, question: str, error_msg: str) -> ProcessingResult:
        """Create error result for failed processing"""
        return ProcessingResult(
            answer=f"I apologize, but I encountered an error while processing this question. Please try again later.",
            confidence=0.0,
            model="error",
            processing_time=0.0,
            validation={
                "error": error_msg,
                "chunks_found": 0,
                "cache_hit": False
            },
            relevant_chunks=[],
            metadata={
                "question": question,
                "error": error_msg,
                "processing_steps": ["error"]
            }
        )

# ============================================================================
