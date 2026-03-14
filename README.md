# Agent Knowledge Distiller

Extract and distill the best knowledge from AI agent memories stored in Qdrant into a curated **golden collection**.

This service is focused on **conversation-memory distillation**:
- scoring and classifying memories
- enriching retained memories
- re-embedding gold memories
- deleting noise from the source collection
- preserving operational state, snapshots, and reports

It is a standalone distillation service and should be understood independently from any separate project-indexing system.

## Features

- Read memories from Qdrant source collection
- Full distill mode and incremental distill mode
- Rule-based scoring with optional LLM-assisted enrichment
- Re-embed retained gold memories
- Delete noise memories from the source collection
- Write curated output to separate golden collection
- Persist distill state to `snapshots/last-distill.json`
- Generate operational summary to `snapshots/distill-operational-summary.json`
- Snapshot/report/progress artifacts for operational visibility

## Project Structure

```text
src/
  index.ts                 # CLI entry
  services/
    qdrant.service.ts      # Qdrant data access + state/report helpers
    scorer.service.ts      # Rule-based scoring + classification
    llm-scorer.service.ts  # Optional LLM scoring/enrichment
    distiller.service.ts   # Main orchestration
    embedding.service.ts   # Re-embed support
  types/
    index.ts               # Shared interfaces/types
  utils/
    config.ts              # Env + defaults

snapshots/                 # Runtime artifacts (gitignored)
```

## Requirements

- Node.js 18+
- Qdrant running at `localhost:6333`
- source collection available (default from env/config)

## Setup

By default this project is expected to live under:

```bash
~/Work/projects/agent-knowledge-distiller
```

Setup locally:

```bash
cd /Users/mrcagents/Work/projects/agent-knowledge-distiller
npm install
```

Shell scripts support runtime path override via:
- `DISTILLER_DIR`
- `PROJECT_WORKSPACE_ROOT`

## Environment

Create `.env` if needed:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=mrc_bot_memory
GOLDEN_COLLECTION=agent_golden_knowledge
SNAPSHOT_DIR=./snapshots
LLM_BASE_URL=http://localhost:8317/v1
LLM_API_KEY=proxypal-local
LLM_MODEL=gpt-5
LLM_SCORING_ENABLED=false
```

## Commands

### Build
```bash
npm run build
```

### Status
```bash
npm run status
# or
node dist/index.js status
```

### Full distill
```bash
npm run distill
# or
npx ts-node src/index.ts distill --dry-run --agent trader
```

### Incremental distill
```bash
npm run incremental
# or
npx ts-node src/index.ts incremental --llm --snapshot --max-per-agent 1500
```

### Report / snapshot
```bash
npm run report
npm run snapshot
```

## CLI Summary

```text
Usage: agent-knowledge-distiller [command]

Commands:
  distill         FULL: re-score + enrich + re-embed ALL memories
  incremental     INCREMENTAL: process only undistilled/new memories
  status          Show collection health and distill state
  snapshot        Create backup snapshot

Common options:
  --agent <name>      Process specific agent
  --min-score <n>     Minimum quality score
  --max-per-agent <n> Max memories to process per agent
  --dry-run           Preview without writing changes
  --snapshot          Create snapshot after run
  --llm               Enable LLM scoring/enrichment when supported
  --rule-only         Force rule-based mode
```

## Runtime Artifacts

Artifacts are written under `snapshots/`.

Important files:
- `last-distill.json`
  - latest persisted distill state
- `distill-operational-summary.json`
  - generated summary of state + recent operational artifacts
- `distill-*.log`
  - distill run logs
- `delta-reembed-report-*.json`
- `delta-reembed-report-*.md`
- `reembed-progress-latest.json`
- `reembed-progress-history.jsonl`
- `*.snapshot`

These are runtime artifacts and are normally gitignored.

## Status Output

`status` now reports:
- source collection count
- distilled vs undistilled count
- golden collection count
- last distill state
- recent operational artifacts

This is intended to make cron/runtime health easier to inspect quickly.

## Safety Notes

- Golden collection remains separate from source collection
- Distill state is persisted for operational continuity
- Snapshots and reports are preserved as runtime artifacts
- Use `--dry-run` before applying changes when validating behavior

## Validation Flow

```bash
npm run build
npm run status
npx ts-node src/index.ts distill --dry-run --agent trader
npx ts-node src/index.ts incremental --dry-run
```
