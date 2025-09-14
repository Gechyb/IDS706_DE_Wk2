# Python Image
FROM python:3.11-slim

# Set up the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Copy requirement file
COPY requirements.txt /app/requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . /app/

# Default command: run test
# CMD ["/bin/bash"]
CMD ["pytest", "-vv", "--cov=data_analysis", "test_data_analysis.py"]
