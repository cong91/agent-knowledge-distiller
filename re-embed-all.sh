#!/bin/bash
# Re-embed ALL memories with qwen3-embedding:0.6b
# This replaces old mxbai-embed-large vectors with qwen3 vectors

set -e
export HOME="/Users/mrcagents"
export PATH="/opt/homebrew/bin:$HOME/.nvm/versions/node/v24.13.1/bin:/usr/local/bin:$PATH"

DISTILLER_DIR="$HOME/.openclaw/workspace/projects/agent-knowledge-distiller"
LOG_FILE="$DISTILLER_DIR/snapshots/re-embed-$(date +%Y-%m-%d).log"

cd "$DISTILLER_DIR"

echo "========================================" >> "$LOG_FILE"
echo "Re-embed ALL: $(date)" >> "$LOG_FILE"
echo "Model: qwen3-embedding:0.6b" >> "$LOG_FILE"
echo "Node: $(node --version)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

TS_NODE="$DISTILLER_DIR/node_modules/.bin/ts-node"

# Create snapshot backup FIRST
echo "Creating backup snapshot..." >> "$LOG_FILE"
"$TS_NODE" src/re-embed-all.ts >> "$LOG_FILE" 2>&1

echo "Completed: $(date)" >> "$LOG_FILE"
