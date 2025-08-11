# ============================================================================

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.llm_service import LLMService

class TestLLMService:
    """Test suite for LLM service"""
    
    @pytest.fixture
    def llm_service(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            return LLMService()
    
    @pytest.mark.asyncio
    async def test_answer_generation(self, llm_service):
        """Test answer generation with context"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer based on provided context."
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        with patch.object(llm_service.client.chat.completions, 'create', return_value=mock_response):
            context_chunks = [
                {
                    'content': 'Sample context about insurance policies.',
                    'similarity': 0.85,
                    'chunk_id': 'test_chunk'
                }
            ]
            
            result = await llm_service.answer_question(
                question="What is covered by this policy?",
                context_chunks=context_chunks,
                document_url="https://example.com/test.pdf"
            )
            
            assert 'answer' in result
            assert 'confidence' in result
            assert 'model' in result
            assert result['confidence'] > 0

# ============================================================================
