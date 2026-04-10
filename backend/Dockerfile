# Use official Python image
FROM python:3.11-slim

# Install system dependencies for OpenCV and Git LFS
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Initialize Git LFS globally
RUN git lfs install --skip-repo

# Set work directory
WORKDIR /app

# Copy Git metadata for LFS to work
COPY .git .git
COPY .gitattributes .gitattributes

# Download LFS files
RUN git lfs pull --include="results/*.h5" || true

# Copy requirements
COPY requirements.txt ./
COPY requirements.system.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8001

# Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
