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

backup_distill_artifacts() {
  local backup_dir="$HOME/Work/Data"
  local snapshot_path=""
  local slotdb_path=""
  local zip_output=""
  local tmp_dir=""

  echo "Backup block started: $(date)" >> "$LOG_FILE"

  if ! mkdir -p "$backup_dir"; then
    echo "[WARN] Backup skipped: cannot create backup dir: $backup_dir" >> "$LOG_FILE"
    return 0
  fi

  snapshot_path="$(ls -1t "$DISTILLER_DIR"/snapshots/*.snapshot 2>/dev/null | head -n 1 || true)"
  if [ -n "$snapshot_path" ] && [ -f "$snapshot_path" ]; then
    echo "Snapshot selected: $snapshot_path" >> "$LOG_FILE"
  else
    snapshot_path=""
    echo "[WARN] No snapshot found in $DISTILLER_DIR/snapshots/*.snapshot" >> "$LOG_FILE"
  fi

  if [ -n "${OPENCLAW_SLOTDB_DIR:-}" ]; then
    if [ -d "$OPENCLAW_SLOTDB_DIR" ]; then
      slotdb_path="$OPENCLAW_SLOTDB_DIR"
    else
      echo "[WARN] OPENCLAW_SLOTDB_DIR is set but missing: $OPENCLAW_SLOTDB_DIR" >> "$LOG_FILE"
    fi
  fi

  if [ -z "$slotdb_path" ] && [ -d "$HOME/.openclaw/SlotDB" ]; then
    slotdb_path="$HOME/.openclaw/SlotDB"
  fi

  if [ -z "$slotdb_path" ] && [ -d "$HOME/.openclaw/workspace/SlotDB" ]; then
    slotdb_path="$HOME/.openclaw/workspace/SlotDB"
  fi

  if [ -n "$slotdb_path" ]; then
    echo "SlotDB selected: $slotdb_path" >> "$LOG_FILE"
  else
    echo "[WARN] SlotDB not found (checked: OPENCLAW_SLOTDB_DIR, ~/.openclaw/SlotDB, ~/.openclaw/workspace/SlotDB)" >> "$LOG_FILE"
  fi

  tmp_dir="$(mktemp -d /tmp/distill-backup.XXXXXX 2>/dev/null || true)"
  if [ -z "$tmp_dir" ] || [ ! -d "$tmp_dir" ]; then
    echo "[WARN] Backup skipped: cannot create temp dir" >> "$LOG_FILE"
    return 0
  fi

  if [ -n "$snapshot_path" ]; then
    if ! cp "$snapshot_path" "$tmp_dir/"; then
      echo "[WARN] Failed to copy snapshot into temp dir" >> "$LOG_FILE"
    fi
  fi

  if [ -n "$slotdb_path" ]; then
    if ! cp -R "$slotdb_path" "$tmp_dir/SlotDB"; then
      echo "[WARN] Failed to copy SlotDB into temp dir" >> "$LOG_FILE"
    fi
  fi

  {
    echo "generated_at=$(date -Iseconds)"
    echo "snapshot_path=${snapshot_path:-<none>}"
    echo "slotdb_path=${slotdb_path:-<none>}"
  } > "$tmp_dir/backup-manifest.txt"

  zip_output="$backup_dir/distill-backup-$(date +%Y%m%d-%H%M%S).zip"
  if (cd "$tmp_dir" && zip -rq "$zip_output" .); then
    echo "ZIP output: $zip_output" >> "$LOG_FILE"
  else
    echo "[WARN] Failed to create zip at: $zip_output" >> "$LOG_FILE"
  fi

  rm -rf "$tmp_dir" || true
  echo "Backup block finished: $(date)" >> "$LOG_FILE"
  return 0
}

echo "========================================" >> "$LOG_FILE"
echo "Distill run: $(date)" >> "$LOG_FILE"
echo "Node: $(node --version) @ $(which node)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Use local ts-node from node_modules (most reliable)
TS_NODE="$DISTILLER_DIR/node_modules/.bin/ts-node"

# Single Source of Truth: read memory runtime config from ~/.openclaw/openclaw.json
OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
if [ -f "$OPENCLAW_CONFIG" ]; then
  eval "$(python3 - <<'PY'
import json
from pathlib import Path
p=Path('/Users/mrcagents/.openclaw/openclaw.json')
obj=json.loads(p.read_text())
cfg=obj.get('plugins',{}).get('entries',{}).get('agent-smart-memo',{}).get('config',{})
qdrant_collection=cfg.get('qdrantCollection','mrc_bot')
embed_model=cfg.get('embedModel','qwen3-embedding:0.6b')
embed_dims=cfg.get('embedDimensions',1024)
print(f'export QDRANT_COLLECTION="{qdrant_collection}"')
print(f'export EMBED_MODEL="{embed_model}"')
print(f'export EMBEDDING_MODEL="{embed_model}"')
print(f'export EMBEDDING_DIMENSIONS="{embed_dims}"')
PY
)"
else
  # Safe fallback
  export QDRANT_COLLECTION="mrc_bot"
  export EMBED_MODEL="qwen3-embedding:0.6b"
  export EMBEDDING_MODEL="qwen3-embedding:0.6b"
  export EMBEDDING_DIMENSIONS="1024"
fi

# Run incremental distill with LLM scoring
"$TS_NODE" src/index.ts incremental --llm --snapshot --max-per-agent 1500 >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
echo "Status after distill:" >> "$LOG_FILE"
"$TS_NODE" src/index.ts status >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
backup_distill_artifacts || true

echo "Completed: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep only last 7 days of logs
find "$DISTILLER_DIR/snapshots" -name "distill-*.log" -mtime +7 -delete 2>/dev/null
