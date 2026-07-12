# Переменные для удобства настройки
IMAGE_NAME = speakver-app
CONTAINER_NAME = speakver_container
PORT = 8501

.PHONY: build run up down stop logs clean help

help: ## Показать справку по командам
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Собрать чистый Docker-образ
	docker build -t $(IMAGE_NAME) .

run: ## Запустить контейнер через чистый Docker с монтированием папок
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		-v $$(pwd)/datasets:/app/datasets \
		-v $$(pwd)/reports:/app/reports \
		-v $$(pwd)/experiments:/app/experiments \
		-v $$(pwd)/debug_audio:/app/debug_audio \
		--restart unless-stopped \
		$(IMAGE_NAME)

up: ## Собрать и запустить проект через docker-compose
	docker-compose up --build -d

down: ## Остановить и удалить контейнеры, созданные через docker-compose
	docker-compose down

stop: ## Остановить и удалить контейнер, запущенный через чистый Docker (make run)
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs: ## Посмотреть логи запущенного контейнера
	docker logs -f $(CONTAINER_NAME)

clean: ## Удалить неиспользуемые Docker-ресурсы (очистка кэша)
	docker system prune -f
