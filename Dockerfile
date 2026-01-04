# Dockerfile for Mohler YouTube Semantic Search API
# Optimized for Google Cloud Run deployment

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY .env.template .
COPY server/ ./server/

# Create necessary directories
RUN mkdir -p server/data server/static

# Cloud Run expects the app to listen on $PORT
# The server will be started from the server directory
WORKDIR /app/server

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')" || exit 1

# Run with gunicorn for production
# - Workers: 2 (Cloud Run recommends 1-2 for most workloads)
# - Threads: 8 (good balance for I/O bound operations)
# - Timeout: 60s (generous for API calls to OpenAI/Pinecone)
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 60 app:app
