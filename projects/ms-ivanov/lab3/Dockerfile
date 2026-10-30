# Retrieval service (MCP over streamable HTTP) for the papers index.
# Built for CPU: queries are embedded one at a time, no GPU is needed.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

WORKDIR /app

COPY source/requirements.txt requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY source ./source
COPY configs ./configs

RUN useradd --create-home --uid 10001 app && mkdir /models && chown app /models
USER app

EXPOSE 8765
CMD ["python", "-m", "source", "mcp", "--transport", "streamable-http", \
     "--host", "0.0.0.0", "--port", "8765"]
