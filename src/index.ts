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
  .description('Distill agent memories — keep gold, delete noise, protect distilled')
  .version('2.0.0');

program
  .command('distill')
  .description('FULL: re-score ALL memories (expensive, use rarely)')
  .option('--agent <name>', 'Process specific agent')
  .option('--min-score <n>', 'Min quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Preview without changes', false)
  .option('--snapshot', 'Create snapshot after', false)
  .option('--llm', 'Use LLM scoring', false)
  .option('--rule-only', 'Force rule-based only', false)
  .action(async (options) => {
    setupEnv(options);
    const config = buildDistillConfig({
      agents: options.agent ? [String(options.agent)] : DEFAULT_AGENTS,
      minScore: options.minScore, maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun, createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });
    const report = await distillerService.distill(config);
    printReport(report, Boolean(options.dryRun));
  });

program
  .command('incremental')
  .description('INCREMENTAL: score only NEW memories, delete noise (daily cron)')
  .option('--agent <name>', 'Process specific agent')
  .option('--min-score <n>', 'Min quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Preview without changes', false)
  .option('--snapshot', 'Create snapshot after', false)
  .option('--llm', 'Use LLM scoring', false)
  .option('--rule-only', 'Force rule-based only', false)
  .action(async (options) => {
    setupEnv(options);
    const config = buildDistillConfig({
      agents: options.agent ? [String(options.agent)] : DEFAULT_AGENTS,
      minScore: options.minScore, maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun, createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });
    const report = await distillerService.incrementalDistill(config);
    printReport(report, Boolean(options.dryRun));
  });

program
  .command('status')
  .description('Show collection health and distill state')
  .action(async () => {
    const sourceCount = await qdrantService.getCount(qdrantService.sourceCollectionName);
    const goldenCount = await qdrantService.getCount(qdrantService.goldenCollectionName).catch(() => 0);
    const distilledCount = await qdrantService.getDistilledCount();
    const state = await qdrantService.loadState();

    console.log(chalk.cyan('\n=== Distill Status ==='));
    console.log(`Source (${qdrantService.sourceCollectionName}): ${sourceCount} points`);
    console.log(`  ├── distilled=true (protected): ${distilledCount}`);
    console.log(`  └── undistilled (needs processing): ${sourceCount - distilledCount}`);
    console.log(`Golden backup (${qdrantService.goldenCollectionName}): ${goldenCount} points`);

    if (state) {
      console.log(chalk.yellow('\nLast distill:'));
      console.log(`  Date: ${state.lastDistillDate}`);
      console.log(`  Gold kept: ${state.goldCount}`);
      console.log(`  Noise deleted: ${state.noiseDeleted}`);
      console.log(`  Source: ${state.sourceCountBefore} → ${state.sourceCountAfter}`);
      console.log(`  Total ever distilled: ${state.totalDistilledEver}`);
    } else {
      console.log(chalk.yellow('\nNo previous distill state found (first run pending)'));
    }
  });

program
  .command('snapshot')
  .description('Create backup snapshot of source collection')
  .action(async () => {
    const path = await qdrantService.createSnapshot(qdrantService.sourceCollectionName);
    console.log(chalk.green(`✅ Snapshot: ${path}`));
  });

program
  .command('report')
  .description('Quick preview — dry-run rule-based scoring')
  .option('--agent <name>', 'Process specific agent')
  .action(async (options) => {
    const agents = options.agent ? [String(options.agent)] : DEFAULT_AGENTS;
    const report = await distillerService.buildReport(agents);
    printReport(report, true);
  });

program.parseAsync(process.argv).catch((err: unknown) => {
  console.error(chalk.red(`❌ ${err instanceof Error ? err.message : String(err)}`));
  process.exit(1);
});

function setupEnv(opts: Record<string, unknown>) {
  if (opts.llm) process.env.LLM_SCORING_ENABLED = 'true';
  if (opts.ruleOnly) process.env.LLM_SCORING_ENABLED = 'false';
}

function parseNumber(v: string): number {
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`Invalid number: ${v}`);
  return n;
}

function printReport(report: DistillReport, dryRun: boolean): void {
  console.log(chalk.cyan('\n=== Distill Report ==='));
  console.log(`Time: ${report.timestamp}`);
  console.log(`Mode: ${report.mode?.toUpperCase() || (dryRun ? 'DRY RUN' : 'WRITE')}`);
  console.log(`Processed: ${report.totalProcessed}`);
  console.log(`Gold (kept): ${report.totalKept}`);
  console.log(`Noise (discarded): ${report.totalDiscarded}`);
  if (report.totalMarkedDistilled != null) console.log(`Marked distilled: ${report.totalMarkedDistilled}`);
  if (report.totalNoiseDeleted != null) console.log(`Noise deleted: ${report.totalNoiseDeleted}`);

  for (const [agent, s] of Object.entries(report.byAgent)) {
    console.log(chalk.yellow(`\n[${agent}] kept=${s.kept}, noise=${s.noiseCount || 0}, total=${s.processed}`));
    for (const m of s.topMemories) {
      console.log(`  (${m.score}) [${m.category}] ${m.text}`);
    }
  }
  if (report.snapshotCreated) console.log(chalk.green(`\nSnapshot: ${report.snapshotCreated}`));
}
