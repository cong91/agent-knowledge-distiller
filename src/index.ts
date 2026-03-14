#!/usr/bin/env node
import chalk from 'chalk';
import { Command } from 'commander';
import { DistillReport } from './types';
import { DistillerService } from './services/distiller.service';
import { QdrantService } from './services/qdrant.service';
import { buildDistillConfig, DEFAULT_AGENTS } from './utils/config';

const program = new Command();
const qdrantService = new QdrantService();
const distillerService = new DistillerService(qdrantService);

program
  .name('agent-knowledge-distiller')
  .description('Distill agent memories — score, enrich, re-embed, clean')
  .version('3.0.0');

program
  .command('distill')
  .description('FULL: re-score + enrich + re-embed ALL memories')
  .option('--agent <name>', 'Process specific agent')
  .option('--min-score <n>', 'Min quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Preview without changes', false)
  .option('--snapshot', 'Create snapshot after', false)
  .option('--llm', 'Use LLM scoring + enrichment', false)
  .option('--rule-only', 'Force rule-based only (no enrichment)', false)
  .action(async (options) => {
    setupEnv(options);
    const config = buildDistillConfig({
      agents: options.agent ? [String(options.agent)] : DEFAULT_AGENTS,
      minScore: options.minScore, maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun, createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });
    printReport(await distillerService.distill(config), Boolean(options.dryRun));
  });

program
  .command('incremental')
  .description('INCREMENTAL: score + enrich + re-embed only NEW memories (daily cron)')
  .option('--agent <name>', 'Process specific agent')
  .option('--min-score <n>', 'Min quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Preview without changes', false)
  .option('--snapshot', 'Create snapshot after', false)
  .option('--llm', 'Use LLM scoring + enrichment', false)
  .option('--rule-only', 'Force rule-based only (no enrichment)', false)
  .action(async (options) => {
    setupEnv(options);
    const config = buildDistillConfig({
      agents: options.agent ? [String(options.agent)] : DEFAULT_AGENTS,
      minScore: options.minScore, maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun, createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });
    printReport(await distillerService.incrementalDistill(config), Boolean(options.dryRun));
  });

program
  .command('status')
  .description('Show collection health and distill state')
  .action(async () => {
    const sourceCount = await qdrantService.getCount(qdrantService.sourceCollectionName);
    const goldenCount = await qdrantService.getCount(qdrantService.goldenCollectionName).catch(() => 0);
    const distilledCount = await qdrantService.getDistilledCount();
    const state = await qdrantService.loadState();
    const operational = await qdrantService.getOperationalSummary();

    console.log(chalk.cyan('\n=== Distill Status ==='));
    console.log(`Source (${qdrantService.sourceCollectionName}): ${sourceCount} points`);
    console.log(`  ├── distilled (protected + re-embedded): ${distilledCount}`);
    console.log(`  └── undistilled (pending): ${sourceCount - distilledCount}`);
    console.log(`Golden backup (${qdrantService.goldenCollectionName}): ${goldenCount} points`);

    if (state) {
      console.log(chalk.yellow('\nLast distill:'));
      console.log(`  Date: ${state.lastDistillDate}`);
      console.log(`  Gold: ${state.goldCount}, Noise deleted: ${state.noiseDeleted}`);
      console.log(`  Source: ${state.sourceCountBefore} → ${state.sourceCountAfter}`);
      console.log(`  Total distilled ever: ${state.totalDistilledEver}`);
    } else {
      console.log(chalk.yellow('\nNo previous state (first run pending)'));
    }

    if (operational.latestArtifacts.length > 0) {
      console.log(chalk.yellow('\nOperational artifacts:'));
      for (const item of operational.latestArtifacts.slice(0, 8)) {
        console.log(`  - [${item.type}] ${item.name} (${item.modifiedAt})`);
      }
    }
  });

program.command('snapshot').description('Create backup snapshot')
  .action(async () => {
    const p = await qdrantService.createSnapshot(qdrantService.sourceCollectionName);
    console.log(chalk.green(`✅ Snapshot: ${p}`));
  });

program.parseAsync(process.argv).catch((err: unknown) => {
  console.error(chalk.red(`❌ ${err instanceof Error ? err.message : String(err)}`));
  process.exit(1);
});

function setupEnv(o: Record<string, unknown>) {
  if (o.llm) process.env.LLM_SCORING_ENABLED = 'true';
  if (o.ruleOnly) process.env.LLM_SCORING_ENABLED = 'false';
}

function parseNumber(v: string): number {
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`Invalid number: ${v}`);
  return n;
}

function printReport(r: DistillReport, dryRun: boolean): void {
  console.log(chalk.cyan('\n=== Distill Report ==='));
  console.log(`Time: ${r.timestamp} | Mode: ${r.mode?.toUpperCase() ?? (dryRun ? 'DRY' : 'WRITE')}`);
  console.log(`Processed: ${r.totalProcessed} | Gold: ${r.totalKept} | Noise: ${r.totalDiscarded}`);
  if (r.totalMarkedDistilled != null) console.log(`Marked distilled: ${r.totalMarkedDistilled}`);
  if (r.totalReembedded != null) console.log(`Re-embedded vectors: ${r.totalReembedded}`);
  if (r.totalNoiseDeleted != null) console.log(`Noise deleted: ${r.totalNoiseDeleted}`);

  for (const [agent, s] of Object.entries(r.byAgent)) {
    console.log(chalk.yellow(`\n[${agent}] gold=${s.kept} noise=${s.noiseCount ?? 0} total=${s.processed}`));
    for (const m of s.topMemories) {
      console.log(`  (${m.score}) [${m.category}] ${m.text}`);
      if (m.enrichedText) console.log(chalk.gray(`    → enriched: ${m.enrichedText}`));
    }
  }
  if (r.snapshotCreated) console.log(chalk.green(`\nSnapshot: ${r.snapshotCreated}`));
}
