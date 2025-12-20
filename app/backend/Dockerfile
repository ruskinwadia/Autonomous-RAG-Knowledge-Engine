FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pdfplumber/pypdf if needed
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY app/backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
