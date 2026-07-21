IMAGE_NAME = speakver-app
CONTAINER_NAME = speakver_container
PORT = 7860

# ============================================================
# Host directories (can be overridden)
# Example:
# make run DATASET_DIR=/mnt/vibravox RUNS_DIR=/mnt/models
# ============================================================

DATASET_DIR ?= $(PWD)/datasets
RUNS_DIR ?= $(PWD)/runs
REPORTS_DIR ?= $(PWD)/reports
DEBUG_AUDIO_DIR ?= $(PWD)/debug_audio

# ============================================================
# Container directories
# ============================================================

CONTAINER_DATASET_DIR = /app/datasets
CONTAINER_RUNS_DIR = /app/runs
CONTAINER_REPORTS_DIR = /app/reports
CONTAINER_DEBUG_AUDIO_DIR = /app/debug_audio

.PHONY: help build run stop shell logs clean rebuild remove

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker build -t $(IMAGE_NAME) .

run: ## Run container with GPU support
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p $(PORT):7860 \
		-v $(DATASET_DIR):$(CONTAINER_DATASET_DIR) \
		-v $(RUNS_DIR):$(CONTAINER_RUNS_DIR) \
		-v $(REPORTS_DIR):$(CONTAINER_REPORTS_DIR) \
		-v $(DEBUG_AUDIO_DIR):$(CONTAINER_DEBUG_AUDIO_DIR) \
		-e DATASET_ROOT=$(CONTAINER_DATASET_DIR) \
		-e RUNS_DIR=$(CONTAINER_RUNS_DIR)/speaker_train \
		-e REPORT_DIR=$(CONTAINER_REPORTS_DIR) \
		-e DEBUG_AUDIO_DIR=$(CONTAINER_DEBUG_AUDIO_DIR) \
		--restart unless-stopped \
		$(IMAGE_NAME)

stop: ## Stop and remove container
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

shell: ## Open bash inside container
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs: ## Follow application logs
	docker logs -f $(CONTAINER_NAME)

remove: ## Remove Docker image
	-docker rmi $(IMAGE_NAME)

rebuild: stop build run ## Rebuild image and restart container

clean: ## Remove unused Docker resources
	docker system prune -f