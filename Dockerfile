FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps some Python packages (pyarrow, cryptography, duckdb, etc.) may need to build/run
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Cloud Run sets $PORT (default 8080) — Streamlit must bind to it and to 0.0.0.0
ENV PORT=8080
EXPOSE 8080

WORKDIR /app/src

# --server.headless keeps Streamlit from trying to open a browser / show the welcome prompt
# --server.address 0.0.0.0 is required so Cloud Run's proxy can reach the container
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
