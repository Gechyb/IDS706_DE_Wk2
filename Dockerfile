# Python Image
FROM python:3.11-slim

# Set up the working directory inside the container
WORKDIR /app

# Copy requirement file
COPY requirements.txt /app/requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . /app/

# Default command: run test
CMD ["/bin/bash"]