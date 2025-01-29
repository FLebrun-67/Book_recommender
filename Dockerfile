# Use a lightweight Python image
FROM python:3.10-slim

# Install necessary tools
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

# Install deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh || { echo "Failed to install Deta CLI"; exit 1; }

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt $HOME/app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Expose the port used by Streamlit
EXPOSE $PORT

# Command to run the Streamlit app
CMD ["streamlit", "run", "--server.port=7860", "app.py"]