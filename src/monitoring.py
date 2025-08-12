# ============================================================================

import asyncio
import logging
import time
from typing import Dict, Any, List
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """System performance monitoring and metrics collection"""
    
    def __init__(self):
        self.request_history = deque(maxlen=1000)  # Keep last 1000 requests
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "total_questions_processed": 0,
            "average_confidence": 0.0
        }
        self.start_time = time.time()
        
        logger.info("✅ Performance monitor initialized")
    
    async def record_request(
        self,
        request_id: str,
        processing_time: float,
        question_count: int,
        avg_confidence: float,
        success: bool = True
    ):
        """Record request metrics"""
        try:
            request_data = {
                "request_id": request_id,
                "timestamp": time.time(),
                "processing_time": processing_time,
                "question_count": question_count,
                "avg_confidence": avg_confidence,
                "success": success
            }
            
            self.request_history.append(request_data)
            
            # Update aggregate metrics
            self.metrics["total_requests"] += 1
            if success:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
            
            self.metrics["total_processing_time"] += processing_time
            self.metrics["total_questions_processed"] += question_count
            
            # Update average confidence (rolling average)
            if self.metrics["successful_requests"] > 0:
                total_confidence = self.metrics["average_confidence"] * (self.metrics["successful_requests"] - 1) + avg_confidence
                self.metrics["average_confidence"] = total_confidence / self.metrics["successful_requests"]
            
        except Exception as e:
            logger.error(f"❌ Failed to record metrics: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate recent performance (last 100 requests)
            recent_requests = list(self.request_history)[-100:]
            recent_successful = [r for r in recent_requests if r["success"]]
            
            recent_metrics = {
                "avg_processing_time": 0.0,
                "avg_confidence": 0.0,
                "success_rate": 0.0,
                "requests_per_minute": 0.0
            }
            
            if recent_successful:
                recent_metrics["avg_processing_time"] = statistics.mean(
                    [r["processing_time"] for r in recent_successful]
                )
                recent_metrics["avg_confidence"] = statistics.mean(
                    [r["avg_confidence"] for r in recent_successful]
                )
            
            if recent_requests:
                recent_metrics["success_rate"] = len(recent_successful) / len(recent_requests)
                
                # Calculate requests per minute based on time span of recent requests
                if len(recent_requests) > 1:
                    time_span = recent_requests[-1]["timestamp"] - recent_requests[0]["timestamp"]
                    if time_span > 0:
                        recent_metrics["requests_per_minute"] = len(recent_requests) / (time_span / 60)
            
            # Performance grade
            performance_grade = self._calculate_performance_grade(recent_metrics)
            
            return {
                "system_info": {
                    "uptime_seconds": uptime,
                    "uptime_formatted": self._format_uptime(uptime),
                    "status": "healthy" if recent_metrics["success_rate"] > 0.8 else "degraded"
                },
                "aggregate_metrics": self.metrics,
                "recent_performance": recent_metrics,
                "performance_grade": performance_grade,
                "request_history_size": len(self.request_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_grade(self, recent_metrics: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        try:
            score = 0
            
            # Success rate (40% weight)
            if recent_metrics["success_rate"] >= 0.95:
                score += 40
            elif recent_metrics["success_rate"] >= 0.9:
                score += 35
            elif recent_metrics["success_rate"] >= 0.8:
                score += 30
            elif recent_metrics["success_rate"] >= 0.7:
                score += 20
            else:
                score += 10
            
            # Processing time (30% weight)
            avg_time = recent_metrics["avg_processing_time"]
            if avg_time <= 5.0:
                score += 30
            elif avg_time <= 10.0:
                score += 25
            elif avg_time <= 15.0:
                score += 20
            elif avg_time <= 20.0:
                score += 15
            else:
                score += 5
            
            # Confidence (30% weight)
            avg_confidence = recent_metrics["avg_confidence"]
            if avg_confidence >= 0.9:
                score += 30
            elif avg_confidence >= 0.8:
                score += 25
            elif avg_confidence >= 0.7:
                score += 20
            elif avg_confidence >= 0.6:
                score += 15
            else:
                score += 5
            
            # Determine grade
            if score >= 90:
                return "A+"
            elif score >= 85:
                return "A"
            elif score >= 80:
                return "A-"
            elif score >= 75:
                return "B+"
            elif score >= 70:
                return "B"
            elif score >= 65:
                return "B-"
            elif score >= 60:
                return "C+"
            elif score >= 55:
                return "C"
            else:
                return "D"
                
        except Exception as e:
            logger.error(f"❌ Grade calculation failed: {e}")
            return "N/A"
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        try:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if seconds > 0 or not parts:
                parts.append(f"{seconds}s")
            
            return " ".join(parts)
        except:
            return "unknown"

# ============================================================================
