#!/usr/bin/env node
import chalk from 'chalk';
import { Command } from 'commander';
import { DistillerService } from './services/distiller.service';
import { QdrantService } from './services/qdrant.service';
import { DistillReport } from "./types";
import { buildDistillConfig, DEFAULT_AGENTS } from './utils/config';

const program = new Command();

const qdrantService = new QdrantService();
const distillerService = new DistillerService(qdrantService);

program
  .name('agent-knowledge-distiller')
  .description('Extract and distill the best knowledge from agent memories')
  .version('2.0.0');

program
  .command('distill')
  .description('FULL distill: score ALL memories, mark gold, delete noise')
  .option('--agent <name>', 'Process specific agent')
  .option('--min-score <n>', 'Minimum quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max golden memories per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Score and report without writing', false)
  .option('--snapshot', 'Create snapshot after distill', false)
  .option('--llm', 'Use LLM scoring', false)
  .option('--rule-only', 'Force rule-based scoring only', false)
  .action(async (options) => {
    setupEnv(options);
    const agents = options.agent ? [String(options.agent)] : DEFAULT_AGENTS;
    const config = buildDistillConfig({
      agents, minScore: options.minScore, maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun, createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });
    const report = await distillerService.distill(config);
    printReport(report, Boolean(options.dryRun));
  });

program
  .command('incremental')
  .description('INCREMENTAL distill: score only NEW memories, delete noise, keep gold')
  .option('--agent <name>', 'Process specific agent')
  .option('--min-score <n>', 'Minimum quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max golden memories per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Score and report without writing', false)
  .option('--snapshot', 'Create snapshot after distill', false)
  .option('--llm', 'Use LLM scoring', false)
  .option('--rule-only', 'Force rule-based scoring only', false)
  .action(async (options) => {
    setupEnv(options);
    const agents = options.agent ? [String(options.agent)] : DEFAULT_AGENTS;
    const config = buildDistillConfig({
      agents, minScore: options.minScore, maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun, createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });
    const report = await distillerService.incrementalDistill(config);
    printReport(report, Boolean(options.dryRun));
  });

program
  .command('snapshot')
  .description('Create a snapshot of source collection')
  .action(async () => {
    const snapshotPath = await qdrantService.createSnapshot(qdrantService.sourceCollectionName);
    console.log(chalk.green(`✅ Snapshot created: ${snapshotPath}`));
  });

program
  .command('report')
  .description('Show statistics and top memories per agent')
  .option('--agent <name>', 'Process specific agent')
  .action(async (options) => {
    const agents = options.agent ? [String(options.agent)] : DEFAULT_AGENTS;
    const report = await distillerService.buildReport(agents);
    printReport(report, true);
  });

program
  .command('status')
  .description('Show collection health — counts, distilled vs undistilled')
  .action(async () => {
    const sourceCount = await qdrantService.getCount(qdrantService.sourceCollectionName);
    const goldenCount = await qdrantService.getCount(qdrantService.goldenCollectionName).catch(() => 0);

    console.log(chalk.cyan('\n=== Distill Status ==='));
    console.log(`Source (${qdrantService.sourceCollectionName}): ${sourceCount} points`);
    console.log(`Golden (${qdrantService.goldenCollectionName}): ${goldenCount} points`);
    console.log(`Noise estimate: ${sourceCount - goldenCount} points`);
    console.log(`\nTip: Run 'incremental --dry-run' to preview what would be cleaned`);
  });

program.parseAsync(process.argv).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(chalk.red(`❌ ${message}`));
  process.exit(1);
});

function setupEnv(options: Record<string, unknown>) {
  if (options.llm) process.env.LLM_SCORING_ENABLED = 'true';
  if (options.ruleOnly) process.env.LLM_SCORING_ENABLED = 'false';
}

function parseNumber(value: string): number {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) throw new Error(`Invalid number: ${value}`);
  return parsed;
}

function printReport(report: DistillReport, dryRun: boolean): void {
  
  console.log(chalk.cyan('\n=== Agent Knowledge Distiller Report ==='));
  console.log(`Timestamp: ${report.timestamp}`);
  console.log(`Mode: ${report.mode?.toUpperCase() || (dryRun ? 'DRY RUN' : 'WRITE')}`);
  console.log(`Total processed: ${report.totalProcessed}`);
  console.log(`Total kept (gold): ${report.totalKept}`);
  console.log(`Total discarded: ${report.totalDiscarded}`);
  if (report.totalMarkedDistilled != null) console.log(`Marked distilled: ${report.totalMarkedDistilled}`);
  if (report.totalNoiseDeleted != null) console.log(`Noise deleted: ${report.totalNoiseDeleted}`);

  for (const [agent, stats] of Object.entries(report.byAgent || {})) {
    const s = stats as any;
    console.log(chalk.yellow(`\n[${agent}] processed=${s.processed}, kept=${s.kept}, noise=${s.noiseCount || 0}`));
    for (const memory of (s.topMemories || [])) {
      console.log(`  - (${memory.score}) [${memory.category}] ${memory.text}`);
    }
  }

  if (report.snapshotCreated) console.log(chalk.green(`\nSnapshot: ${report.snapshotCreated}`));
}
