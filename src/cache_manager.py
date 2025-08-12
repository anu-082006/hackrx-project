# ============================================================================

import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, Optional
import redis.asyncio as redis
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based caching with local fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        self.use_redis = False
        self.max_local_cache_size = 1000
    
    async def initialize(self):
        """Initialize Redis connection with fallback to local cache"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
            
            # Test connection
            await self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis cache initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using local cache: {e}")
            self.use_redis = False
    
    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        try:
            if self.use_redis and self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                return self.local_cache.get(cache_key)
                
        except Exception as e:
            logger.warning(f"⚠️ Cache get failed: {e}")
        
        return None
    
    async def cache_result(
        self, 
        cache_key: str, 
        result: Dict[str, Any], 
        ttl: int = 3600
    ) -> bool:
        """Cache result with TTL"""
        try:
            if self.use_redis and self.redis_client:
                serialized_data = json.dumps(result, default=str)
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            else:
                # Local cache with size limit
                if len(self.local_cache) >= self.max_local_cache_size:
                    # Remove oldest entries (simple FIFO)
                    old_keys = list(self.local_cache.keys())[:100]
                    for key in old_keys:
                        del self.local_cache[key]
                
                self.local_cache[cache_key] = result
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Cache set failed: {e}")
            return False
    
    def generate_cache_key(self, document_url: str, question: str) -> str:
        """Generate consistent cache key"""
        combined = f"{document_url}:{question}"
        return f"hackrx:qa:{hashlib.md5(combined.encode()).hexdigest()}"
    
    def generate_document_cache_key(self, document_url: str) -> str:
        """Generate cache key for document processing"""
        return f"hackrx:doc:{hashlib.md5(document_url.encode()).hexdigest()}"
    
    async def optimize_cache(self):
        """Optimize cache performance"""
        try:
            if not self.use_redis:
                # For local cache, just limit size
                if len(self.local_cache) > self.max_local_cache_size * 1.2:
                    # Remove 20% of oldest entries
                    keys_to_remove = list(self.local_cache.keys())[:int(len(self.local_cache) * 0.2)]
                    for key in keys_to_remove:
                        del self.local_cache[key]
                    logger.info(f"🧹 Local cache optimized, removed {len(keys_to_remove)} entries")
        
        except Exception as e:
            logger.warning(f"⚠️ Cache optimization failed: {e}")
    
    async def close(self):
        """Close cache connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            self.local_cache.clear()
            logger.info("✅ Cache manager closed")
        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")

# ============================================================================
