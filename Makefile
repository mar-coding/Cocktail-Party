# =============================================================================
# Cocktail Party - Voice AI
# =============================================================================
# Domain: cocktail-party.site
# CI/CD: Push to main triggers auto-deployment
# =============================================================================

APP_DIR := local-voice-ai-agent

.PHONY: help setup build dev dev-down dev-logs prod prod-down prod-logs up down restart logs status clean

help:
	@echo "Cocktail Party - Voice AI"
	@echo "========================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup       - Full setup (certs + model + build)"
	@echo "  make build       - Build Docker images"
	@echo ""
	@echo "Development:"
	@echo "  make dev         - Start development mode"
	@echo "  make dev-down    - Stop development containers"
	@echo "  make dev-logs    - View development logs"
	@echo ""
	@echo "Production:"
	@echo "  make prod        - Start production mode (checks IS_PROD in .env)"
	@echo "  make prod-down   - Stop production containers"
	@echo "  make prod-logs   - View production logs"
	@echo ""
	@echo "Common:"
	@echo "  make up          - Alias for 'make prod'"
	@echo "  make down        - Stop all containers"
	@echo "  make restart     - Restart containers"
	@echo "  make logs        - View all logs"
	@echo "  make status      - Container status"
	@echo "  make clean       - Remove containers and volumes"

# =============================================================================
# Setup
# =============================================================================

setup:
	@cd $(APP_DIR) && $(MAKE) setup

build:
	@cd $(APP_DIR) && $(MAKE) build

# =============================================================================
# Development
# =============================================================================

dev:
	@cd $(APP_DIR) && $(MAKE) dev

dev-down:
	@cd $(APP_DIR) && $(MAKE) dev-down

dev-logs:
	@cd $(APP_DIR) && $(MAKE) dev-logs

# =============================================================================
# Production
# =============================================================================

prod:
	@cd $(APP_DIR) && $(MAKE) prod

prod-down:
	@cd $(APP_DIR) && $(MAKE) prod-down

prod-logs:
	@cd $(APP_DIR) && $(MAKE) prod-logs

# =============================================================================
# Common
# =============================================================================

up:
	@cd $(APP_DIR) && $(MAKE) up

down:
	@cd $(APP_DIR) && $(MAKE) down

restart:
	@cd $(APP_DIR) && $(MAKE) restart

logs:
	@cd $(APP_DIR) && $(MAKE) logs

status:
	@cd $(APP_DIR) && $(MAKE) status

clean:
	@cd $(APP_DIR) && $(MAKE) clean
