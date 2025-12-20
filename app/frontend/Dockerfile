FROM python:3.11-slim

WORKDIR /app

COPY app/frontend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app/frontend/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
