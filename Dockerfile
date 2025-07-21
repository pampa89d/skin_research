FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Установка зависимостей системы
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pages/ pages/
COPY models/ models/
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "pages/app.py", "--server.address=0.0.0.0"]
