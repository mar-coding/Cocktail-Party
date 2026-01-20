# =============================================================================
# Voice AI Makefile - Simple deployment for local and production
# =============================================================================

.PHONY: help setup setup-model certs build up down restart logs shell clean dev prod status

# Default target
help:
	@echo "Voice AI - Makefile Commands"
	@echo "============================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Full setup: generate certs + pull model + build"
	@echo "  make setup-model  - Pull the Ollama model only"
	@echo "  make certs        - Generate SSL certificates"
	@echo "  make build        - Build Docker images"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start in development mode (with source mounting)"
	@echo "  make dev-down     - Stop development containers"
	@echo "  make dev-logs     - View development logs"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start in production mode"
	@echo "  make prod-down    - Stop production containers"
	@echo "  make prod-logs    - View production logs"
	@echo ""
	@echo "Common:"
	@echo "  make up           - Alias for 'make prod'"
	@echo "  make down         - Stop all containers"
	@echo "  make restart      - Restart production containers"
	@echo "  make logs         - View all logs (follow mode)"
	@echo "  make status       - Show container status"
	@echo "  make shell        - Open shell in voice-app container"
	@echo "  make clean        - Remove containers and volumes"
	@echo ""
	@echo "Environment Variables:"
	@echo "  OLLAMA_MODEL      - Model to use (default: gemma3:4b)"
	@echo "  HF_TOKEN          - HuggingFace token for private models"

# =============================================================================
# Setup Commands
# =============================================================================

setup: certs build setup-model
	@echo "Setup complete! Run 'make prod' or 'make dev' to start."

setup-model:
	@echo "Pulling Ollama model..."
	docker compose --profile setup up ollama-pull

certs:
	@echo "Generating SSL certificates..."
	@mkdir -p certs
	@if [ ! -f certs/cert.pem ]; then \
		./generate-certs.sh; \
	else \
		echo "Certificates already exist. Delete certs/ to regenerate."; \
	fi

build:
	@echo "Building Docker images..."
	docker compose build

# =============================================================================
# Development Commands
# =============================================================================

dev: certs
	@echo "Starting in development mode..."
	docker compose --profile dev up -d ollama voice-app-dev
	@echo ""
	@echo "Voice AI running at: https://localhost:7860"
	@echo "Run 'make dev-logs' to view logs"

dev-down:
	docker compose --profile dev down voice-app-dev

dev-logs:
	docker compose --profile dev logs -f voice-app-dev

dev-restart: dev-down dev

# =============================================================================
# Production Commands
# =============================================================================

prod: certs
	@echo "Starting in production mode..."
	docker compose up -d ollama voice-app
	@echo ""
	@echo "Voice AI running at: https://localhost:7860"
	@echo "Run 'make prod-logs' to view logs"

prod-down:
	docker compose down voice-app

prod-logs:
	docker compose logs -f voice-app

prod-restart:
	docker compose down voice-app
	docker compose build voice-app
	docker compose up -d voice-app

# =============================================================================
# Common Commands
# =============================================================================

up: prod

down:
	docker compose --profile dev down

restart: prod-restart

logs:
	docker compose logs -f

status:
	@echo "Container Status:"
	@docker compose ps
	@echo ""
	@echo "Ollama Models:"
	@docker exec ollama ollama list 2>/dev/null || echo "Ollama not running"

shell:
	docker exec -it voice-app /bin/bash

shell-ollama:
	docker exec -it ollama /bin/bash

# =============================================================================
# Cleanup Commands
# =============================================================================

clean:
	@echo "Stopping and removing containers..."
	docker compose --profile dev --profile setup down -v
	@echo "Removing certificates..."
	rm -rf certs/
	@echo "Clean complete."

clean-volumes:
	@echo "Removing Docker volumes (this will delete cached models)..."
	docker volume rm ollama_models voice_ai_hf_cache voice_ai_kokoro_cache 2>/dev/null || true
	@echo "Volumes removed."

# =============================================================================
# Quick Rebuild (for development)
# =============================================================================

rebuild: build prod-restart
	@echo "Rebuild complete!"
