# =============================================================================
# LibrarAI - Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
#
# Build:
#   docker build -t librar_ai:latest .
#
# Run standalone:
#   docker run -it --rm \
#     -v $(pwd)/data:/app/data:ro \
#     -v $(pwd)/indices:/app/indices \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     librar_ai:latest python -m src.cli.ask_library "¿Qué es BB84?"
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim as runtime

# Labels
LABEL maintainer="LibrarAI"
LABEL description="RAG system for quantum computing and physics library"
LABEL version="1.0.0"

# Create non-root user
RUN groupadd -r librar_ai && useradd -r -g librar_ai librar_ai

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=librar_ai:librar_ai src/ ./src/
COPY --chown=librar_ai:librar_ai config/ ./config/
COPY --chown=librar_ai:librar_ai setup.py ./
COPY --chown=librar_ai:librar_ai README.md ./

# Create necessary directories
RUN mkdir -p data indices logs outputs && \
    chown -R librar_ai:librar_ai /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Default to INFO logging in JSON format for production
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json

# Switch to non-root user
USER librar_ai

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Default command (interactive mode)
CMD ["python", "-m", "src.cli.ask_library", "--interactive"]
