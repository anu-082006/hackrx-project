# ============================================================================

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import openai
import os
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMService:
    """OpenAI GPT service for intelligent question answering"""
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment")
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4"
        self.fallback_model = "gpt-3.5-turbo"
        self.max_tokens = 500
        self.temperature = 0.1  # Low temperature for consistent, factual responses
        
        logger.info("✅ LLM service initialized")
    
    async def answer_question(
        self, 
        question: str, 
        context_chunks: List[Dict], 
        document_url: str
    ) -> Dict[str, Any]:
        """Generate answer using GPT with context"""
        start_time = time.time()
        
        try:
            # Prepare context from relevant chunks
            context = self._prepare_context(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(question, context, document_url)
            
            # Try GPT-4 first, fallback to GPT-3.5
            try:
                response = await self._make_llm_request(prompt, self.model)
            except Exception as e:
                logger.warning(f"⚠️ GPT-4 failed, trying fallback: {e}")
                response = await self._make_llm_request(prompt, self.fallback_model)
            
            processing_time = time.time() - start_time
            
            # Extract answer and validate
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(
                answer, context_chunks, response.usage
            )
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model": response.model,
                "processing_time": processing_time,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "context_chunks_used": len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"❌ LLM request failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "confidence": 0.0,
                "model": "error",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def _make_llm_request(self, prompt: str, model: str) -> Any:
        """Make request to OpenAI API with retries"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant that answers questions based on provided context. Always provide accurate, specific answers based on the given information. If the information is not in the context, clearly state that."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Rate limit hit, waiting {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise e
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                raise e
        
        raise Exception("All retry attempts failed")
    
    def _prepare_context(self, context_chunks: List[Dict]) -> str:
        """Prepare context from relevant chunks"""
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to top 5 chunks
            similarity = chunk.get('similarity', 0.0)
            content = chunk.get('content', '').strip()
            
            if content:
                context_parts.append(f"[Context {i}] (Relevance: {similarity:.2f})\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str, document_url: str) -> str:
        """Create optimized prompt for question answering"""
        prompt = f"""Based on the following context from a document, please answer the question accurately and specifically.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be specific and include relevant details from the context
3. If the answer is not in the context, clearly state "The provided context does not contain information to answer this question"
4. For numerical values, dates, or specific terms, quote them exactly as they appear in the context
5. Keep your answer concise but complete
6. If there are conditions or exceptions mentioned in the context, include them in your answer

ANSWER:"""
        
        return prompt
    
    def _calculate_confidence(
        self, 
        answer: str, 
        context_chunks: List[Dict], 
        usage: Any
    ) -> float:
        """Calculate confidence score for the answer"""
        base_confidence = 0.5
        
        # Boost confidence if we have good context
        if context_chunks:
            avg_similarity = sum(chunk.get('similarity', 0) for chunk in context_chunks) / len(context_chunks)
            base_confidence += avg_similarity * 0.3
        
        # Boost confidence for longer, more detailed answers
        if len(answer) > 50:
            base_confidence += 0.1
        if len(answer) > 100:
            base_confidence += 0.1
        
        # Reduce confidence for uncertain phrases
        uncertain_phrases = [
            "not sure", "don't know", "unclear", "might be", "could be",
            "not contain information", "not in the context"
        ]
        
        for phrase in uncertain_phrases:
            if phrase in answer.lower():
                base_confidence -= 0.3
                break
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, base_confidence))

# ============================================================================
