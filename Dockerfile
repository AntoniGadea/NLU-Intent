# Dockerfile

# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app will run on
EXPOSE 8080

# The command to run the application using uvicorn
# It tells uvicorn to run the 'app' instance from the 'serve.py' file
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]