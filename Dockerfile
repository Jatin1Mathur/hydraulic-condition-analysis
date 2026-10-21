FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models
ENV HYDRAULIC_MODEL_PATH=/app/models/condition_models.joblib

EXPOSE 8000
CMD ["uvicorn", "hydraulic.api:app", "--host", "0.0.0.0", "--port", "8000"]
