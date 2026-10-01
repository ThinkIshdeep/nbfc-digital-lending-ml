FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY artifacts ./artifacts
COPY data/processed ./data/processed
COPY frontend ./frontend

RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "nbfc_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
