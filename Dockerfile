FROM python:3.10-slim

WORKDIR /app

# Важно: обновляем pip и ставим нужный typing-extensions заранее
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "typing-extensions>=4.10.0"

# Теперь PyTorch и остальное ставится нормально
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers fastapi uvicorn[standard] pillow requests sentencepiece protobuf python-multipart

COPY api_server_siglip.py api_server.py

EXPOSE 8000
CMD ["python", "api_server.py"]