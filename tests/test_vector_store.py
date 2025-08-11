# ============================================================================

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from src.vector_store import VectorStoreManager

class TestVectorStore:
    """Test suite for vector store operations"""
    
    @pytest.fixture
    def vector_store(self):
        vs = VectorStoreManager()
        vs.use_pinecone = False  # Force local mode for testing
        return vs
    
    @pytest.mark.asyncio
    async def test_store_and_search_chunks(self, vector_store):
        """Test storing and searching document chunks"""
        # Sample chunks with embeddings
        chunks = [
            {
                'chunk_id': 'chunk1',
                'content': 'This is about insurance policies and coverage.',
                'embedding': np.random.rand(384).tolist(),
                'metadata': {'confidence': 0.9}
            },
            {
                'chunk_id': 'chunk2', 
                'content': 'Medical expenses and hospital coverage details.',
                'embedding': np.random.rand(384).tolist(),
                'metadata': {'confidence': 0.8}
            }
        ]
        
        # Store chunks
        doc_url = 'https://example.com/test.pdf'
        success = await vector_store.store_document_chunks(doc_url, chunks)
        assert success
        
        # Search for relevant chunks
        relevant_chunks = await vector_store.search_relevant_chunks(
            query='insurance coverage',
            document_url=doc_url,
            top_k=2,
            confidence_threshold=0.1  # Low threshold for testing
        )
        
        assert len(relevant_chunks) > 0
        assert all('similarity' in chunk for chunk in relevant_chunks)

# ============================================================================
