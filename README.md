# SpeakVer 🎙️

**SpeakVer** — система верификации и идентификации говорящего по голосу, построенная на архитектуре **ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification)**.

Проект реализует полный цикл разработки системы Speaker Verification:

- подготовка и предобработка аудиоданных;
- обучение модели;
- оценка качества с использованием специализированных метрик;
- генерация подробных отчетов;
- веб-интерфейс на **Gradio** для интерактивной проверки модели.

---

## 🔥 Основные возможности

- **ECAPA-TDNN** для извлечения Speaker Embeddings.
- **Автоматическая подготовка датасета** и построение структуры данных.
- **Гибкое обучение модели** с возможностью загрузки чекпоинтов.
- **Автоматический подбор оптимального порога** по EER.
- **Генерация подробных отчетов** с графиками ROC, Confusion Matrix и распределением косинусного сходства.
- **Gradio Web UI** для демонстрации работы модели.
- **Динамическая загрузка моделей** без перезапуска приложения.
- **Автоматическое архивирование отчетов** после генерации.

---

## 📂 Структура проекта

```text
SpeakVer/
│
├── app.py                     # Точка входа Gradio-приложения
├── train.py                   # Обучение модели
├── report.py                  # Генерация отчета
├── download_data.py           # Подготовка датасета
│
├── datasets/                  # Датасеты
├── reports/                   # Сгенерированные отчеты
├── archives/                  # ZIP-архивы отчетов
├── runs/                      # Эксперименты и чекпоинты
├── config/                    # Конфигурация проекта
│
├── src/
│   │
│   ├── config.py              # Глобальная конфигурация проекта
|   ├── __init__.py 
│   │
│   ├── data/
|   |   ├── __init__.py
│   │   ├── audio_preprocessor.py
│   │   ├── baseline_report.py
│   │   ├── metrics.py
│   │   ├── pair_builder.py
│   │   └── plotter.py
│   │
│   ├── model/
|   |   ├── __init__.py
│   │   ├── aamsoftmax.py
│   │   └── embedding_extractor.py
│   │
│   ├── speaker_verifier/
|   |   ├── __init__.py
│   │   └── speaker_verifier.py
│   │
│   ├── train/
|   |   ├── __init__.py
│   │   ├── speaker_dataset.py
│   │   └── trainer.py
│   │
│   ├── gradio/
|   |   ├── __init__.py
│   │   ├── state.py           # Глобальное состояние приложения
│   │   │
│   │   ├── services/
|   |   |   ├── __init__.py
│   │   │   ├── auth_service.py
│   │   │   ├── model_service.py
│   │   │   └── report_service.py
│   │   │
│   │   ├── ui/
|   |   |   ├── __init__.py
│   │   │   ├── layout.py
│   │   │   ├── verification_tab.py
│   │   │   └── report_tab.py
│   │   │
│   │   ├── utils/
|   |   |   ├── __init__.py
│   │   │   ├── archive.py
│   │   │   ├── model_utils.py
│   │   │   ├── report_parser.py
│   │   │   └── report_utils.py
│   │   │
│   │   └── templates/
│   │       └── report.md
│   │
│   └── utils/
|       ├── __init__.py
│       └── utils.py
│
├── requirements.txt
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── pyproject.toml
└── README_GRADIO_INTERFACE.md
└── README_DOCKER_MAKE_COMPOSE.md
└── README.md
```

---

## 🛠️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/AlexBorealis/SpeakVer.git

cd SpeakVer
```

---

### 2. Создание виртуального окружения

```bash
python -m venv .venv
```

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

---

### 3. Установка зависимостей

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

При необходимости установите системные библиотеки:

- ffmpeg
- libsndfile

---

## 🚀 Использование

### 1. Подготовка датасета

```bash
python download_data.py
```

---

### 2. Обучение модели

```bash
python train.py
```

вызов справки по сервису

```bash
python train.py --help
```

После обучения автоматически сохраняются:

- веса модели;
- конфигурация эксперимента;
- история обучения.

---

### 3. Генерация отчета

```bash
python report.py
```

вызов справки по сервису

```bash
python report.py --help
```

Отчет содержит:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- EER
- оптимальный Threshold
- ROC Curve
- Confusion Matrix
- Similarity Distribution
- JSON с метриками
- CSV со всеми парами

Все отчеты автоматически сохраняются в каталог **reports/**.

---

### 4. Запуск Gradio-приложения

```bash
python app.py
```

вызов справки по сервису

```bash
python app.py --help
```

После запуска приложение будет доступно по адресу [http://localhost:7860](http://localhost:7860)

---

## 🌐 Возможности Gradio-приложения

Интерфейс состоит из двух независимых вкладок.

## 🎙 Verification

Позволяет:

- выбрать эксперимент;
- выбрать чекпоинт;
- динамически загрузить модель;
- настроить порог косинусного сходства;
- сравнить две аудиозаписи;
- получить:

  - SAME / DIFFERENT;
  - Confidence;
  - Cosine Similarity.

---

## 📊 Report

Позволяет:

- выбрать модель;
- выбрать тестовый датасет;
- включить или отключить балансировку пар;
- указать имя отчета;
- автоматически сформировать отчет;
- автоматически создать ZIP-архив;
- скачать архив одним нажатием;
- просмотреть краткую сводку результатов непосредственно в интерфейсе.

---

## 🧠 Используемая модель

В проекте используется архитектура **ECAPA-TDNN**, которая расширяет классический X-vector подход за счет:

- SE-Res2Blocks;
- Channel Attention;
- Multi-layer Feature Aggregation;
- Attentive Statistics Pooling;
- нормализованных Speaker Embeddings.

Полученные эмбеддинги сравниваются с использованием **Cosine Similarity**.

---

## 📈 Метрики оценки

Во время тестирования рассчитываются:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Equal Error Rate (EER)
- оптимальный порог классификации
- Confusion Matrix
- ROC Curve
- распределение Cosine Similarity.

---

## 👤 Автор

**AlexBorealis**
GitHub: <https://github.com/AlexBorealis/SpeakVer>
