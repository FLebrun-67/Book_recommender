# Use a lightweight Python image
FROM python:3.11-slim

# Install only necessary system packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for Hugging Face Spaces
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy requirements.txt first to leverage Docker cache
COPY --chown=user requirements.txt $HOME/app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . $HOME/app

# Ensure artifacts directory exists (important for your models)
RUN mkdir -p $HOME/app/artifacts && \
    mkdir -p $HOME/app/data

# Health check to ensure the app starts correctly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Expose the port
EXPOSE $PORT

# Command to run the Streamlit app with Hugging Face specific settings
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]