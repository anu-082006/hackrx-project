# ============================================================================

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.document_processor import AdvancedDocumentProcessor

class TestDocumentProcessor:
    """Test suite for document processing"""
    
    @pytest.fixture
    def processor(self):
        return AdvancedDocumentProcessor()
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, processor):
        """Test complete document processing pipeline"""
        # Mock successful processing
        with patch.object(processor, 'extract_with_all_methods') as mock_extract:
            mock_extract.return_value = {
                'text': 'Sample document text for testing purposes.',
                'confidence': 0.95,
                'method': 'pymupdf'
            }
            
            result = await processor.process_document('https://example.com/test.pdf')
            
            assert 'chunks' in result
            assert 'metadata' in result
            assert len(result['chunks']) > 0
    
    @pytest.mark.asyncio
    async def test_smart_chunking(self, processor):
        """Test intelligent text chunking"""
        sample_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = await processor.create_smart_chunks(
            text=sample_text,
            metadata={'confidence': 0.9},
            chunk_size=50
        )
        
        assert len(chunks) > 0
        assert all('chunk_id' in chunk for chunk in chunks)
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)

# ============================================================================
