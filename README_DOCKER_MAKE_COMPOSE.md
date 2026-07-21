# 🐳 Запуск в [Docker](Dockerfile)

Проект полностью поддерживает запуск внутри Docker-контейнера. Это позволяет использовать одинаковое окружение на любой машине без необходимости устанавливать все зависимости вручную.

Контейнер основан на образе **NVIDIA CUDA Runtime**, поэтому может использовать GPU для ускорения инференса и обучения.

Используемый базовый образ:

```dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
```

В контейнер автоматически устанавливаются:

- Python
- PyTorch
- SpeechBrain
- CUDA Runtime
- FFmpeg
- libsndfile
- все зависимости из [requirements.txt](requirements.txt)

При запуске автоматически стартует веб-приложение Gradio.

---

# 📁 Монтируемые каталоги

Для удобства работы данные хранятся вне контейнера и подключаются как Docker Volumes.

| Каталог проекта | Каталог внутри контейнера | Назначение |
|-----------------|---------------------------|------------|
| `datasets/` | `/app/datasets` | Датасеты |
| `runs/` | `/app/runs` | Чекпоинты моделей |
| `reports/` | `/app/reports` | Отчеты |
| `debug_audio/` | `/app/debug_audio` | Тестовые аудиофайлы |

Это означает, что после пересборки контейнера все данные сохраняются.

---

# ⚙️ Конфигурация

Все параметры приложения вынесены в файл окружения.

Например:

```text
config/envs/.env
```

В нем задаются:

- пути к данным;
- учетные записи пользователей;
- параметры запуска;
- другие переменные окружения.

Пример:

```env
USER1_LOGIN=alex
USER1_PASSWORD=password

USER2_LOGIN=tester
USER2_PASSWORD=password

USER3_LOGIN=demo
USER3_PASSWORD=password
```

---

# 🚀 Запуск через [Docker](Dockerfile)

## Сборка образа

```bash
make build
```

или

```bash
docker build -t speakver-app .
```

---

## Запуск контейнера

```bash
make run
```

или

```bash
docker run \
    --gpus all \
    -p 127.0.0.1:7860:7860 \
    --env-file config/envs/docker.env \
    speakver-app
```

После запуска приложение будет доступно по адресу

```
http://localhost:7860
```

---

## Просмотр логов

```bash
make logs
```

---

## Подключение внутрь контейнера

```bash
make shell
```

---

## Остановка контейнера

```bash
make stop
```

---

## Пересборка контейнера

Если изменился код приложения:

```bash
make rebuild
```

---

# 🐳 Использование [Docker Compose](docker-compose.yml)

Для более удобного управления сервисом используется [docker-compose.yml](docker-compose.yml).

## Сборка

```bash
make build-compose
```

или

```bash
docker compose build
```

---

## Запуск

```bash
make run-compose
```

или

```bash
docker compose up -d
```

---

## Просмотр логов

```bash
make logs-compose
```

---

## Вход внутрь контейнера

```bash
make shell-compose
```

---

## Остановка

```bash
make stop-compose
```

---

## Полная пересборка

```bash
make rebuild-compose
```

---

# 🔧 [Makefile](Makefile)

Для удобства вся работа с Docker автоматизирована через Makefile.

Доступные команды:

| Команда | Назначение |
|----------|------------|
| `make help` | список всех команд |
| `make build` | сборка Docker образа |
| `make run` | запуск контейнера |
| `make stop` | остановка контейнера |
| `make rebuild` | пересборка контейнера |
| `make logs` | просмотр логов |
| `make shell` | открыть Bash внутри контейнера |
| `make remove` | удалить Docker образ |
| `make clean` | очистить неиспользуемые Docker ресурсы |

Для Docker Compose:

| Команда | Назначение |
|----------|------------|
| `make build-compose` | сборка compose |
| `make run-compose` | запуск compose |
| `make stop-compose` | остановка compose |
| `make rebuild-compose` | пересборка compose |
| `make logs-compose` | просмотр логов |
| `make shell-compose` | Bash внутри compose-контейнера |

---

# 🎮 Использование GPU

Контейнер поддерживает работу с NVIDIA GPU.

Для этого необходимы:

- установленный драйвер NVIDIA;
- Docker Engine;
- NVIDIA Container Toolkit.

Проверить доступность GPU можно командой:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu24.04 nvidia-smi
```

Если выводится информация о видеокарте, значит контейнер сможет использовать GPU.

---

# 🌐 Публикация приложения

По умолчанию контейнер публикует сервис только на localhost

```
127.0.0.1:7860
```

Это сделано специально для повышения безопасности.

Для предоставления доступа внешним пользователям рекомендуется использовать:

- ngrok;
- Cloudflare Tunnel;
- обратный прокси (Nginx);
- VPS с собственным доменом.

---

# 🔐 Авторизация

Веб-интерфейс поддерживает встроенную авторизацию Gradio.

Учетные записи задаются через файл окружения:

```env
USER1_LOGIN=alex
USER1_PASSWORD=password

USER2_LOGIN=tester
USER2_PASSWORD=password

USER3_LOGIN=demo
USER3_PASSWORD=password
```

При запуске приложения пользователю необходимо пройти авторизацию перед использованием сервиса.

---