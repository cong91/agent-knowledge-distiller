#!/usr/bin/env node
import chalk from 'chalk';
import { Command } from 'commander';
import { DistillerService } from './services/distiller.service';
import { QdrantService } from './services/qdrant.service';
import { buildDistillConfig, DEFAULT_AGENTS } from './utils/config';

const program = new Command();

const qdrantService = new QdrantService();
const distillerService = new DistillerService(qdrantService);

program
  .name('agent-knowledge-distiller')
  .description('Extract and distill the best knowledge from agent memories')
  .version('1.0.0');

program
  .command('distill')
  .description('Extract and score golden knowledge from agent memories')
  .option('--agent <name>', 'Process specific agent (trader/fullstack/scrum/assistant)')
  .option('--min-score <n>', 'Minimum quality score (default: 60)', parseNumber)
  .option('--max-per-agent <n>', 'Max golden memories per agent (default: 100)', parseNumber)
  .option('--dry-run', 'Score and report without writing to golden collection', false)
  .option('--snapshot', 'Create snapshot after distill', false)
  .option('--llm', 'Use LLM scoring (requires GEMINI_API_KEY)', false)
  .option('--rule-only', 'Force rule-based scoring only', false)
  .action(async (options) => {
    if (options.llm) {
      process.env.LLM_SCORING_ENABLED = 'true';
    }
    if (options.ruleOnly) {
      process.env.LLM_SCORING_ENABLED = 'false';
    }

    const agents = options.agent ? [String(options.agent)] : DEFAULT_AGENTS;

    const config = buildDistillConfig({
      agents,
      minScore: options.minScore,
      maxPerAgent: options.maxPerAgent,
      dryRun: options.dryRun,
      createSnapshot: options.snapshot,
      forceRuleOnly: Boolean(options.ruleOnly),
    });

    const report = await distillerService.distill(config);
    printReport(report, Boolean(options.dryRun));
  });

program
  .command('snapshot')
  .description('Create a snapshot of the golden collection')
  .action(async () => {
    await qdrantService.createGoldenCollection();
    const snapshotPath = await qdrantService.createSnapshot(qdrantService.goldenCollectionName);
    console.log(chalk.green(`✅ Snapshot created: ${snapshotPath}`));
  });

program
  .command('report')
  .description('Show statistics and top memories per agent')
  .option('--agent <name>', 'Process specific agent (trader/fullstack/scrum/assistant)')
  .action(async (options) => {
    const agents = options.agent ? [String(options.agent)] : DEFAULT_AGENTS;
    const report = await distillerService.buildReport(agents);
    printReport(report, true);
  });

program.parseAsync(process.argv).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(chalk.red(`❌ ${message}`));
  process.exit(1);
});

function parseNumber(value: string): number {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`Invalid number: ${value}`);
  }
  return parsed;
}

function printReport(report: Awaited<ReturnType<DistillerService['distill']>>, dryRun: boolean): void {
  console.log(chalk.cyan('\n=== Agent Knowledge Distiller Report ==='));
  console.log(`Timestamp: ${report.timestamp}`);
  console.log(`Mode: ${dryRun ? 'DRY RUN' : 'WRITE'}`);
  console.log(`Total processed: ${report.totalProcessed}`);
  console.log(`Total kept: ${report.totalKept}`);
  console.log(`Total discarded: ${report.totalDiscarded}`);

  for (const [agent, stats] of Object.entries(report.byAgent)) {
    console.log(chalk.yellow(`\n[${agent}] processed=${stats.processed}, kept=${stats.kept}`));
    for (const memory of stats.topMemories) {
      console.log(`  - (${memory.score}) [${memory.category}] ${memory.text}`);
    }
  }

  if (report.snapshotCreated) {
    console.log(chalk.green(`\nSnapshot: ${report.snapshotCreated}`));
  }
}
