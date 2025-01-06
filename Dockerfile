# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port that your app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
