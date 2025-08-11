# ============================================================================

#!/bin/bash

# HackRx 6.0 Deployment Script
echo "🚀 Starting HackRx 6.0 Deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one with your API keys."
    echo "Required variables:"
    echo "- OPENAI_API_KEY"
    echo "- PINECONE_API_KEY (optional)"
    echo "- PINECONE_ENVIRONMENT (optional)"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run tests (if available)
if [ -f "tests/test_main.py" ]; then
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
fi

# Start the application
echo "🔄 Starting FastAPI application..."
if [ "$1" == "production" ]; then
    echo "🏭 Production mode"
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
elif [ "$1" == "docker" ]; then
    echo "🐳 Docker mode"
    docker-compose up --build -d
    echo "✅ Application started with Docker Compose"
    echo "📊 Check health: curl http://localhost:8000/health"
    echo "📈 Check metrics: curl http://localhost:8000/metrics"
else
    echo "🔧 Development mode"
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi

# ============================================================================
