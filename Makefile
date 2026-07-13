IMAGE_NAME = speakver-app
CONTAINER_NAME = speakver_container
PORT = 7860

.PHONY: build run up down stop logs clean help

help: ## Показать справку по командам
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Собрать Docker-образ с CUDA и Python 3.12
	docker build -t $(IMAGE_NAME) .

run: ## Запустить контейнер с поддержкой GPU и монтированием томов
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p $(PORT):$(PORT) \
		-v $$(pwd)/datasets:/app/datasets \
		-v $$(pwd)/reports:/app/reports \
		-v $$(pwd)/experiments:/app/experiments \
		-v $$(pwd)/debug_audio:/app/debug_audio \
		--restart unless-stopped \
		$(IMAGE_NAME)

up: ## Собрать и запустить проект с GPU через docker-compose
	docker-compose up --build -d

down: ## Остановить и удалить контейнеры docker-compose
	docker-compose down

stop: ## Остановить одиночный контейнер (после make run)
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

shell: ## Войти в терминал запущенного контейнера
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs: ## Посмотреть живые логи Gradio приложения
	docker logs -f $(CONTAINER_NAME)

clean: ## Очистить кэш Docker системных слоев
	docker system prune -f
