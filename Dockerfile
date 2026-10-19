FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY healthtrack ./healthtrack
COPY static ./static
COPY data ./data

# Run as a non-root user; logs/ is the only writable location.
RUN useradd --system --uid 10001 app && mkdir -p logs && chown -R app:app /app/logs
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["python", "-m", "healthtrack"]
