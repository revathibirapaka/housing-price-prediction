# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy all contents into the container
COPY . /app

# Install dependencies from requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Run the main script
CMD ["python", "main.py"]


