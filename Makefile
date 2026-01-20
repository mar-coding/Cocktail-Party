# =============================================================================
# Cocktail Party - Voice AI
# =============================================================================
# Domain: cocktail-party.site
# CI/CD: Push to main triggers auto-deployment
# =============================================================================

APP_DIR := local-voice-ai-agent

.PHONY: help setup build dev prod down logs status clean

help:
	@echo "Cocktail Party - Voice AI"
	@echo "========================="
	@echo ""
	@echo "  make setup   - Full setup (certs + model + build)"
	@echo "  make build   - Build Docker images"
	@echo "  make dev     - Start development mode"
	@echo "  make prod    - Start production mode"
	@echo "  make down    - Stop containers"
	@echo "  make logs    - View logs"
	@echo "  make status  - Container status"
	@echo "  make clean   - Remove containers and volumes"

setup:
	@cd $(APP_DIR) && $(MAKE) setup

build:
	@cd $(APP_DIR) && $(MAKE) build

dev:
	@cd $(APP_DIR) && $(MAKE) dev

prod:
	@cd $(APP_DIR) && $(MAKE) prod

down:
	@cd $(APP_DIR) && $(MAKE) down

logs:
	@cd $(APP_DIR) && $(MAKE) logs

status:
	@cd $(APP_DIR) && $(MAKE) status

clean:
	@cd $(APP_DIR) && $(MAKE) clean