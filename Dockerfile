# Use official Python image as base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY src/ /app/src
COPY output/model_output /app/output/model_output
COPY dataset/ /app/dataset
COPY config.py /app/
COPY entrypoint.sh /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Set PYTHONPATH so that Python can locate modules under /app (e.g., src/)
ENV PYTHONPATH="/app"

# Copy the entrypoint script into the container and make it executable
RUN chmod +x /app/entrypoint.sh

# Expose ports needed (Streamlit on 8501, FastAPI on 8000)
EXPOSE 8501 8000

# Set a default ENTRYPOINT_MODE (can be overridden at runtime)
ENV ENTRYPOINT_MODE=streamlit

# Use the entrypoint script as the container's entry point
ENTRYPOINT ["bash", "/app/entrypoint.sh"]

# Default command (can be overridden at runtime)
CMD []
