FROM python:3.9-slim

# Install dependencies required for OpenCV, psutil, and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    gcc \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt and install Python packages
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy the rest of the application
COPY . /user/blockage-detection

# Set working directory
WORKDIR /user/blockage-detection

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python3", "main.py"]
