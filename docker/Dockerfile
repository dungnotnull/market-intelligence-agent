FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r agentuser && useradd -r -g agentuser agentuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data output models && \
    chown -R agentuser:agentuser /app

USER agentuser

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

CMD ["python", "-m", "agent.main", "serve", "--host", "0.0.0.0", "--port", "8007"]
