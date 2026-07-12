# Используем официальный легковесный образ Python 3.12
FROM python:3.12-slim

# Устанавливаем системные зависимости для работы со звуком
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Обновляем pip и ставим зависимости (включая setuptools на случай компиляции C-расширений)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Копируем только исходный код, модели по умолчанию и конфиги
COPY src/ ./src/
COPY pretrained_models/ ./pretrained_models/
COPY app.py .

# Создаем пустые папки для автоматического монтирования томов
RUN mkdir -p datasets reports experiments debug_audio

# Открываем порт для приложения
EXPOSE 8501

# Запуск приложения
CMD ["python", "app.py"]
