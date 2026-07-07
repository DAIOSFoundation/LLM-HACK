#!/usr/bin/env bash
# SFASS 2026 레드팀 대시보드 실행
cd "$(dirname "$0")"
exec ./.venv/bin/python src/server.py
