#!/usr/bin/env bash

set -e
set -x

cd "$(dirname "$0")/../app"
uv run mypy .
uv run ruff check .
uv run ruff format . --check
