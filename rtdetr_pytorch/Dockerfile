FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install google-cloud-storage if using GCS, though kagglehub is now primary
RUN pip install --no-cache-dir google-cloud-storage

# Copy the application code
COPY . .

# Set the entrypoint to the wrapper script
ENTRYPOINT ["python", "entrypoint.py"]
