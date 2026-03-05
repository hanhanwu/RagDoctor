FROM python:3.13-slim

# Improve Python runtime behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system dependencies (needed for numpy/pandas wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install uv \
    && uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Railway dynamic port
RUN chmod +x start.sh
CMD ["bash", "start.sh"]