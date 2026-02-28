# Agent Knowledge Distiller

Extract and distill the best knowledge from AI agent memories stored in Qdrant into a curated **golden collection**.

## Features

- Read all memories from Qdrant (`mrc_bot_memory`)
- Rule-based quality scoring (Vietnamese + English signals)
- Category classification by agent/domain
- Keep only top high-quality memories per agent
- Write curated output to separate collection (`agent_golden_knowledge`)
- Snapshot support for backup/restore workflows
- Dry-run mode for safe testing

## Project Structure

```text
src/
  index.ts                 # CLI entry
  services/
    qdrant.service.ts      # Qdrant data access
    scorer.service.ts      # Rule-based scoring + classification
    distiller.service.ts   # Main orchestration
  types/
    index.ts               # Shared interfaces/types
  utils/
    config.ts              # Env + defaults
```

## Requirements

- Node.js 18+
- Qdrant running at `localhost:6333`
- Source collection: `mrc_bot_memory`

## Setup

```bash
cd /Users/mrcagents/.openclaw/workspace/projects/agent-knowledge-distiller
npm install
```

## Environment

Create `.env`:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=mrc_bot_memory
GOLDEN_COLLECTION=agent_golden_knowledge
OPENAI_API_KEY=optional_for_llm_scoring
SNAPSHOT_DIR=./snapshots
```

> `OPENAI_API_KEY` reserved for future LLM scoring (not required in current rule-based version).

## Commands

```bash
npm run build
npm run report
npm run distill
npm run snapshot
```

Or direct CLI usage:

```bash
npx ts-node src/index.ts report
npx ts-node src/index.ts distill --dry-run --agent trader
npx ts-node src/index.ts distill --min-score 70 --max-per-agent 50
npx ts-node src/index.ts snapshot
```

## CLI

```text
Usage: agent-knowledge-distiller [command]

Commands:
  distill         Extract and score golden knowledge from agent memories
  snapshot        Create a snapshot of the golden collection
  report          Show statistics and top memories per agent

Options:
  --agent <name>      Process specific agent (trader/fullstack/scrum/assistant)
  --min-score <n>     Minimum quality score (default: 60)
  --max-per-agent <n> Max golden memories per agent (default: 100)
  --dry-run           Score and report without writing to golden collection
```

## Scoring Rules (Rule-Based v1)

Baseline: `50`

Positive signals (examples):
- `win`, `thắng`, `lợi nhuận` (+15)
- `pattern`, `mẫu` (+10)
- `fix`, `resolved`, `đã sửa` (+10)
- `rule`, `quy tắc`, `luật` (+10)
- `rsi`, `macd`, `sma` (+5)
- `architecture`, `design` (+10)
- `lesson`, `bài học`, `kinh nghiệm` (+15)
- Longer detailed content (+5 / +10)

Negative signals:
- `error` + `500` (-10)
- too short `< 30 chars` (-20)
- `skipping`, `no output` (-15)
- `test` (excluding `backtest`) (-10)

Final score clamped to `[0, 100]`.

## Categories

- `trading_win_pattern`
- `trading_loss_lesson`
- `market_insight`
- `bug_fix_pattern`
- `architecture_decision`
- `code_pattern`
- `process_improvement`
- `project_context`
- `system_rule`
- `noise` (discarded)

## Safety Notes

- Original memories are never deleted.
- Golden collection is separate (`agent_golden_knowledge`).
- Use `--dry-run` before writing.

## Extensibility

- Add/edit scoring signals in `ScorerService`.
- Add future `LLMScorerService` behind feature flag.
- Extend categories via `MemoryCategory` and filtering config.

## Validation Flow

```bash
npm run build
npx ts-node src/index.ts report
npx ts-node src/index.ts distill --dry-run --agent trader
```
