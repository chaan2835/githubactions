# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

# Copy model, API code, and any needed files
COPY model/ model/
COPY . .

# Expose port (for FastAPI)
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
