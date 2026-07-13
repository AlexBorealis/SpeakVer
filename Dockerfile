# Используем официально существующий образ NVIDIA на базе Ubuntu 24.04 с CUDA 12.6
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Отключаем интерактивные диалоги
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем системные аудио-зависимости, компилятор и стандартный python3-pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем список зависимостей
COPY requirements.txt .

# Встроенный в Ubuntu 24.04 Python 3.12 ставит зависимости без конфликтов
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt

# Копируем только логику приложения и модели по умолчанию
COPY src/ ./src/
COPY pretrained_models/ ./pretrained_models/
COPY app.py .

# Создаем директории под автоматическое монтирование внешних томов
RUN mkdir -p datasets reports experiments debug_audio

# Стандартный порт для Gradio
EXPOSE 7860

# Настройка переменных окружения для Gradio внутри Docker
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Запуск вашего Gradio приложения
CMD ["python3", "app.py"]
