#!/usr/bin/env bash
set -euo pipefail

ERROR_LOG="error.log"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_requirements() {
    echo "Checking Python files..."
    python -m py_compile $(git ls-files '*.py')

    echo "Checking Docker installation..."
    command_exists docker || { echo "Docker is required" >&2; exit 1; }

    echo "Checking docker-compose..."
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        echo "docker-compose or 'docker compose' command is required" >&2
        exit 1
    fi
    echo "All checks passed."
}

start_services() {
    echo "Starting services..."
    if command_exists docker-compose; then
        docker-compose up -d
    else
        docker compose up -d
    fi
}

monitor_errors() {
    echo "Monitoring logs. Errors will be written to $ERROR_LOG"
    : > "$ERROR_LOG"
    if command_exists docker-compose; then
        docker-compose logs -f 2>&1 | tee >(grep -i "error" >> "$ERROR_LOG")
    else
        docker compose logs -f 2>&1 | tee >(grep -i "error" >> "$ERROR_LOG")
    fi
}

check_requirements
start_services
monitor_errors
