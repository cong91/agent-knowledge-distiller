#!/bin/bash
# Agent Knowledge Distiller — Daily Incremental Cron
# Schedule: 0 4 * * * (4:00 AM daily)
# Purpose: Score new memories, keep gold, delete noise
# Fixed 2026-03-02: Use absolute paths (cron doesn't load shell profile)

set -e

export HOME="/Users/mrcagents"
# Use Homebrew node (stable) + NVM fallback
export PATH="/opt/homebrew/bin:$HOME/.nvm/versions/node/v24.13.1/bin:$HOME/.bun/bin:/usr/local/bin:$PATH"

DISTILLER_DIR="$HOME/.openclaw/workspace/projects/agent-knowledge-distiller"
LOG_FILE="$DISTILLER_DIR/snapshots/distill-$(date +%Y-%m-%d).log"

# Ensure snapshots dir exists
mkdir -p "$DISTILLER_DIR/snapshots"

cd "$DISTILLER_DIR"

echo "========================================" >> "$LOG_FILE"
echo "Distill run: $(date)" >> "$LOG_FILE"
echo "Node: $(node --version) @ $(which node)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Use local ts-node from node_modules (most reliable)
TS_NODE="$DISTILLER_DIR/node_modules/.bin/ts-node"

# Run incremental distill with LLM scoring
"$TS_NODE" src/index.ts incremental --llm --snapshot --max-per-agent 1500 >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
echo "Status after distill:" >> "$LOG_FILE"
"$TS_NODE" src/index.ts status >> "$LOG_FILE" 2>&1

echo "Completed: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep only last 7 days of logs
find "$DISTILLER_DIR/snapshots" -name "distill-*.log" -mtime +7 -delete 2>/dev/null
