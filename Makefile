# ============================================================
# Configuration
# ============================================================
IMAGE_NAME := speakver-app
CONTAINER_NAME := speakver_container

PORT ?= 7860

ENV_FILE ?= config/envs/.env

# ============================================================
# Host directories
# ============================================================
DATASET_DIR ?= $(PWD)/datasets
RUNS_DIR ?= $(PWD)/runs
REPORTS_DIR ?= $(PWD)/reports
DEBUG_AUDIO_DIR ?= $(PWD)/debug_audio
ARCHIVES_DIR ?= $(PWD)/archives

# ============================================================
# Container directories
# ============================================================
CONTAINER_DATASET_DIR := /app/datasets
CONTAINER_RUNS_DIR := /app/runs
CONTAINER_REPORTS_DIR := /app/reports
CONTAINER_DEBUG_AUDIO_DIR := /app/debug_audio
CONTAINER_ARCHIVES_DIR := /app/archives

# ============================================================
# PHONY
# ============================================================
.PHONY: \
	help \
	build build-compose \
	run run-compose \
	stop stop-compose \
	rebuild rebuild-compose \
	logs logs-compose \
	shell shell-compose \
	remove clean

# ============================================================
# Help
# ============================================================
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Docker
# ============================================================
build: ## Build Docker image
	docker build -t $(IMAGE_NAME) .

run: ## Run container (docker)
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus all \
		--env-file $(ENV_FILE) \
		-p 127.0.0.1:$(PORT):7860 \
		-v $(DATASET_DIR):$(CONTAINER_DATASET_DIR) \
		-v $(RUNS_DIR):$(CONTAINER_RUNS_DIR) \
		-v $(REPORTS_DIR):$(CONTAINER_REPORTS_DIR) \
		-v $(DEBUG_AUDIO_DIR):$(CONTAINER_DEBUG_AUDIO_DIR) \
		-v $(ARCHIVES_DIR):$(CONTAINER_ARCHIVES_DIR) \
		--restart unless-stopped \
		$(IMAGE_NAME)

stop: ## Stop docker container
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

logs: ## Docker logs
	docker logs -f $(CONTAINER_NAME)

shell: ## Bash inside docker container
	docker exec -it $(CONTAINER_NAME) bash

rebuild: stop build run ## Rebuild docker image

# ============================================================
# Docker Compose
# ============================================================
build-compose: ## Build docker compose
	docker compose --env-file $(ENV_FILE) build

run-compose: ## Start docker compose
	docker compose --env-file $(ENV_FILE) up -d

stop-compose: ## Stop docker compose
	docker compose down

logs-compose: ## Docker compose logs
	docker compose logs -f

shell-compose: ## Bash inside compose container
	docker exec -it $(CONTAINER_NAME) bash

rebuild-compose: ## Rebuild docker compose
	docker compose down
	docker compose --env-file $(ENV_FILE) build
	docker compose --env-file $(ENV_FILE) up -d

# ============================================================
# Cleanup
# ============================================================
remove: ## Remove image
	-docker rmi $(IMAGE_NAME)

clean: ## Remove unused docker resources
	docker system prune -f