FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser && mkdir /app/artifacts && chown appuser:appuser /app/artifacts
USER appuser
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn probeflow.reference_api:create_app --factory --host 0.0.0.0 --port \"${PORT:-10000}\" --no-access-log"]

