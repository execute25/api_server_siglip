FROM python:3.10-slim
WORKDIR /app

# ⚡ Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ⚡ Обновляем pip и ставим нужную версию typing-extensions
RUN pip install --upgrade pip && \
    pip install typing-extensions==4.15.0

# ⚡ Устанавливаем PyTorch с поддержкой CUDA 13.0 и остальные зависимости
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir \
    transformers fastapi uvicorn[standard] pillow requests sentencepiece protobuf python-multipart

COPY api_server_siglip.py api_server.py

EXPOSE 8000
CMD ["python", "api_server.py"]
