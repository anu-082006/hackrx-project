# ============================================================================

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pinecone
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Enhanced vector store with Pinecone integration and local fallback"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.local_store = {}  # Fallback local storage
        self.pinecone_index = None
        self.use_pinecone = False
        
    async def initialize(self):
        """Initialize vector store with Pinecone or local fallback"""
        try:
            # Try to initialize Pinecone
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
            index_name = os.getenv('PINECONE_INDEX_NAME', 'hackrx-documents')
            
            if pinecone_api_key and pinecone_env:
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
                
                # Check if index exists, create if not
                if index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=index_name,
                        dimension=384,  # all-MiniLM-L6-v2 dimension
                        metric="cosine"
                    )
                    logger.info(f"✅ Created Pinecone index: {index_name}")
                
                self.pinecone_index = pinecone.Index(index_name)
                self.use_pinecone = True
                logger.info("✅ Pinecone vector store initialized")
            else:
                logger.warning("⚠️ Pinecone not configured, using local vector store")
                
        except Exception as e:
            logger.warning(f"⚠️ Pinecone initialization failed: {e}. Using local fallback")
            self.use_pinecone = False
    
    async def store_document_chunks(self, document_url: str, chunks: List[Dict]) -> bool:
        """Store document chunks in vector database"""
        if not chunks:
            return False
            
        try:
            doc_id = self._get_document_id(document_url)
            
            if self.use_pinecone and self.pinecone_index:
                # Prepare vectors for Pinecone
                vectors = []
                for i, chunk in enumerate(chunks):
                    vector_id = f"{doc_id}_{chunk['chunk_id']}"
                    vectors.append({
                        "id": vector_id,
                        "values": chunk["embedding"],
                        "metadata": {
                            "document_url": document_url,
                            "chunk_id": chunk["chunk_id"],
                            "content": chunk["content"][:1000],  # Truncate for metadata
                            "full_content": chunk["content"],
                            **chunk["metadata"]
                        }
                    })
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    self.pinecone_index.upsert(vectors=batch)
                    
                logger.info(f"✅ Stored {len(chunks)} chunks in Pinecone")
            else:
                # Store locally
                self.local_store[doc_id] = {
                    "document_url": document_url,
                    "chunks": chunks,
                    "embeddings": np.array([chunk["embedding"] for chunk in chunks])
                }
                logger.info(f"✅ Stored {len(chunks)} chunks locally")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store chunks: {e}")
            return False
    
    async def search_relevant_chunks(
        self, 
        query: str, 
        document_url: str, 
        top_k: int = 5,
        confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """Search for relevant chunks using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            doc_id = self._get_document_id(document_url)
            
            if self.use_pinecone and self.pinecone_index:
                # Search in Pinecone
                search_results = self.pinecone_index.query(
                    vector=query_embedding.tolist(),
                    filter={"document_url": {"$eq": document_url}},
                    top_k=top_k,
                    include_metadata=True
                )
                
                relevant_chunks = []
                for match in search_results.matches:
                    if match.score >= confidence_threshold:
                        relevant_chunks.append({
                            "content": match.metadata.get("full_content", match.metadata.get("content", "")),
                            "similarity": match.score,
                            "chunk_id": match.metadata.get("chunk_id"),
                            "metadata": {k: v for k, v in match.metadata.items() 
                                       if k not in ["content", "full_content"]}
                        })
                
            else:
                # Search locally
                if doc_id not in self.local_store:
                    logger.warning(f"⚠️ Document not found in local store: {doc_id}")
                    return []
                
                doc_data = self.local_store[doc_id]
                chunk_embeddings = doc_data["embeddings"]
                
                # Calculate similarities
                similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
                
                # Get top-k results
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                relevant_chunks = []
                for idx in top_indices:
                    if similarities[idx] >= confidence_threshold:
                        chunk = doc_data["chunks"][idx]
                        relevant_chunks.append({
                            "content": chunk["content"],
                            "similarity": float(similarities[idx]),
                            "chunk_id": chunk["chunk_id"],
                            "metadata": chunk["metadata"]
                        })
            
            logger.info(f"🔍 Found {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"❌ Chunk search failed: {e}")
            return []
    
    def _get_document_id(self, document_url: str) -> str:
        """Generate consistent document ID from URL"""
        import hashlib
        return hashlib.md5(document_url.encode()).hexdigest()[:16]
    
    async def close(self):
        """Cleanup vector store resources"""
        try:
            if self.pinecone_index:
                # Pinecone doesn't require explicit closing
                pass
            self.local_store.clear()
            logger.info("✅ Vector store closed")
        except Exception as e:
            logger.error(f"❌ Vector store cleanup failed: {e}")

# ============================================================================
