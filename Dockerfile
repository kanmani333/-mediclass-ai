FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install gunicorn first separately
RUN pip install --no-cache-dir gunicorn==21.2.0

# Install all other packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary folders
RUN mkdir -p database uploads

# Expose port
EXPOSE 8080

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1"]