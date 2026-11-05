FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser && mkdir /app/artifacts && chown appuser:appuser /app/artifacts
USER appuser
EXPOSE 8765
CMD ["uvicorn", "probeflow.reference_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]

