FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers fastapi uvicorn[standard] pillow requests sentencepiece protobuf python-multipart
COPY api_server_siglip.py api_server.py
EXPOSE 8000
CMD ["python", "api_server.py"]
