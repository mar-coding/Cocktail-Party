# =============================================================================
# Cocktail Party - Voice AI
# =============================================================================
# Domain: cocktail-party.site
# CI/CD: Push to main triggers auto-deployment
# =============================================================================

.PHONY: help setup setup-model certs build up down restart logs shell clean dev prod status network-setup

# Ensure networks exist and clean stale containers
network-setup:
	@docker rm -f ollama-pull 2>/dev/null || true
	@docker rm -f voice-app 2>/dev/null || true
	@docker rm -f voice-app-dev 2>/dev/null || true

# Default target
help:
	@echo "Voice AI - Makefile Commands"
	@echo "============================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Full setup: generate certs + pull model + build"
	@echo "  make setup-model  - Pull the Ollama model (Docker Ollama only)"
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
	@echo "LLM Provider Options (set in .env):"
	@echo "  LLM_PROVIDER=ollama     - Use Ollama (Docker or local Mac)"
	@echo "  LLM_PROVIDER=vllm       - Use vLLM via RunPod (no Ollama needed)"
	@echo ""
	@echo "  OLLAMA_HOST examples:"
	@echo "    http://ollama:11434              - Docker Ollama container"
	@echo "    http://host.docker.internal:11434 - Local Mac Ollama"
	@echo "    http://YOUR_VPS_IP:11434         - Remote GPU server"

# =============================================================================
# Setup Commands
# =============================================================================

setup: network-setup certs build setup-model
	@echo "Setup complete! Run 'make prod' or 'make dev' to start."

setup-model:
	@echo "Pulling Ollama model..."
	@docker rm -f ollama-pull 2>/dev/null || true
	docker compose --profile ollama up ollama-pull

certs:
	@echo "Generating SSL certificates..."
	@mkdir -p certs
	@if [ ! -f certs/cert.pem ]; then \
		./scripts/generate-certs.sh; \
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
	@LLM_PROVIDER=$$(grep -E "^LLM_PROVIDER=" .env 2>/dev/null | tail -1 | cut -d= -f2 || echo "ollama"); \
	OLLAMA_HOST=$$(grep -E "^OLLAMA_HOST=" .env 2>/dev/null | tail -1 | cut -d= -f2 || echo "http://ollama:11434"); \
	if [ "$$LLM_PROVIDER" = "vllm" ]; then \
		echo "Using vLLM via RunPod (no Ollama container needed)"; \
		docker compose --profile dev up -d voice-app-dev; \
	elif echo "$$OLLAMA_HOST" | grep -q "ollama:11434"; then \
		echo "Using Docker Ollama container"; \
		docker compose --profile dev --profile ollama up -d voice-app-dev; \
	else \
		echo "Using external Ollama at $$OLLAMA_HOST"; \
		docker compose --profile dev up -d voice-app-dev; \
	fi
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
	@LLM_PROVIDER=$$(grep -E "^LLM_PROVIDER=" .env 2>/dev/null | tail -1 | cut -d= -f2 || echo "ollama"); \
	OLLAMA_HOST=$$(grep -E "^OLLAMA_HOST=" .env 2>/dev/null | tail -1 | cut -d= -f2 || echo "http://ollama:11434"); \
	NEED_OLLAMA="false"; \
	if [ "$$LLM_PROVIDER" = "ollama" ] && echo "$$OLLAMA_HOST" | grep -q "ollama:11434"; then \
		NEED_OLLAMA="true"; \
	fi; \
	if grep -q "^IS_PROD=true" .env 2>/dev/null; then \
		echo "IS_PROD=true detected, using production config (no port mapping)..."; \
		docker network create web_proxy 2>/dev/null || true; \
		if [ "$$NEED_OLLAMA" = "true" ]; then \
			echo "Using Docker Ollama container"; \
			docker compose --profile ollama up -d voice-app; \
		else \
			echo "Using $$LLM_PROVIDER ($$OLLAMA_HOST)"; \
			docker compose up -d voice-app; \
		fi; \
		docker network connect web_proxy voice-app 2>/dev/null || true; \
		echo ""; \
		echo "Voice AI running behind proxy on internal network"; \
	else \
		echo "IS_PROD=false detected, using local config (with port mapping)..."; \
		if [ "$$NEED_OLLAMA" = "true" ]; then \
			echo "Using Docker Ollama container"; \
			docker compose --profile dev --profile ollama up -d voice-app-dev; \
		else \
			echo "Using $$LLM_PROVIDER ($$OLLAMA_HOST)"; \
			docker compose --profile dev up -d voice-app-dev; \
		fi; \
		echo ""; \
		echo "Voice AI running at: https://localhost:7860"; \
	fi
	@echo "Run 'make prod-logs' to view logs"

prod-down:
	@if grep -q "^IS_PROD=true" .env 2>/dev/null; then \
		docker compose down voice-app; \
	else \
		docker compose --profile dev down voice-app-dev; \
	fi

prod-logs:
	@if grep -q "^IS_PROD=true" .env 2>/dev/null; then \
		docker compose logs -f voice-app; \
	else \
		docker compose --profile dev logs -f voice-app-dev; \
	fi

prod-restart:
	@if grep -q "^IS_PROD=true" .env 2>/dev/null; then \
		docker compose down voice-app && \
		docker compose build voice-app && \
		docker compose up -d voice-app; \
	else \
		docker compose --profile dev down voice-app-dev && \
		docker compose --profile dev build voice-app-dev && \
		docker compose --profile dev up -d voice-app-dev; \
	fi

# =============================================================================
# Common Commands
# =============================================================================

up: prod

down:
	docker compose --profile dev --profile ollama down --remove-orphans

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
	docker compose --profile dev --profile ollama down -v
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
