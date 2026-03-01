#!/bin/bash
# Agent Knowledge Distiller — Daily Incremental Cron
# Schedule: 0 4 * * * (4:00 AM daily)
# Purpose: Score new memories, keep gold, delete noise

set -e

export HOME="/Users/mrcagents"
export PATH="$HOME/.nvm/versions/node/v20.18.0/bin:$HOME/.bun/bin:/usr/local/bin:$PATH"

DISTILLER_DIR="$HOME/.openclaw/workspace/projects/agent-knowledge-distiller"
LOG_FILE="$DISTILLER_DIR/snapshots/distill-$(date +%Y-%m-%d).log"

cd "$DISTILLER_DIR"

echo "========================================" >> "$LOG_FILE"
echo "Distill run: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run incremental distill with LLM scoring
npx ts-node src/index.ts incremental --llm --snapshot >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
echo "Status after distill:" >> "$LOG_FILE"
npx ts-node src/index.ts status >> "$LOG_FILE" 2>&1

echo "Completed: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep only last 7 days of logs
find "$DISTILLER_DIR/snapshots" -name "distill-*.log" -mtime +7 -delete 2>/dev/null
