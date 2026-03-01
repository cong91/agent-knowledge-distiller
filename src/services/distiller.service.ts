import { DistillConfig, DistillReport, DistillState, MemoryCategory, ScoredMemory } from '../types';
import { QdrantService } from './qdrant.service';
import { LLMScorerService } from './llm-scorer.service';
import { preFilter, scoreMemory } from './scorer.service';

export class DistillerService {
  constructor(private readonly qdrantService: QdrantService) {}

  /** FULL distill: score ALL memories (ignores state/flags) */
  async distill(config: DistillConfig): Promise<DistillReport> {
    return this.runDistill(config, 'full');
  }

  /** INCREMENTAL distill: score only NEW undistilled memories */
  async incrementalDistill(config: DistillConfig): Promise<DistillReport> {
    return this.runDistill(config, 'incremental');
  }

  private async runDistill(config: DistillConfig, mode: 'full' | 'incremental'): Promise<DistillReport> {
    const report: DistillReport = {
      timestamp: new Date().toISOString(), mode,
      totalProcessed: 0, totalKept: 0, totalDiscarded: 0,
      totalNoiseDeleted: 0, totalMarkedDistilled: 0, byAgent: {},
    };

    const selectedByAgent: Record<string, ScoredMemory[]> = {};
    const noiseByAgent: Record<string, string[]> = {};

    // LLM setup
    const llmBaseUrl = process.env.LLM_BASE_URL || 'http://localhost:8317/v1';
    const llmApiKey = process.env.LLM_API_KEY || 'proxypal-local';
    const llmModel = process.env.LLM_MODEL || 'gemini-2.5-flash';
    const llmEnabled = process.env.LLM_SCORING_ENABLED === 'true' && !config.forceRuleOnly;

    let llmScorer: LLMScorerService | null = null;
    if (llmEnabled) {
      llmScorer = new LLMScorerService(
        { baseUrl: llmBaseUrl, apiKey: llmApiKey, model: llmModel },
        Number.parseInt(process.env.LLM_BATCH_SIZE || '10', 10),
      );
      console.log(`🧠 LLM scoring enabled (${llmModel} via ${llmBaseUrl})`);
    } else {
      console.log('📏 Rule-based scoring (set --llm for LLM)');
    }

    // Load state for incremental
    const prevState = await this.qdrantService.loadState();
    if (mode === 'incremental' && prevState) {
      console.log(`📋 Mode: INCREMENTAL (since ${prevState.lastDistillDate})`);
      console.log(`   Previous run: ${prevState.goldCount} gold, ${prevState.noiseDeleted} noise deleted`);
    } else {
      console.log(`📋 Mode: ${mode.toUpperCase()}${mode === 'incremental' ? ' (first run — no previous state)' : ''}`);
    }

    const beforeCount = await this.qdrantService.getCount(this.qdrantService.sourceCollectionName);
    const distilledCount = await this.qdrantService.getDistilledCount();
    console.log(`📊 Source: ${beforeCount} total, ${distilledCount} distilled, ${beforeCount - distilledCount} undistilled`);

    for (const agent of config.agents) {
      const memories = mode === 'incremental'
        ? await this.qdrantService.getUndistilledMemories(agent)
        : await this.qdrantService.getAllMemories(agent);

      if (memories.length === 0) {
        console.log(`  [${agent}] No ${mode === 'incremental' ? 'new' : ''} memories`);
        continue;
      }

      const filtered = memories.filter(preFilter);
      console.log(`  [${agent}] ${memories.length} → pre-filter: ${filtered.length}`);

      const scored = llmScorer
        ? await llmScorer.scoreBatch(filtered)
        : filtered.map((m) => scoreMemory(m));

      const gold = scored
        .filter((m) => m.qualityScore >= config.minQualityScore)
        .filter((m) => m.category !== 'noise')
        .filter((m) => config.categories.includes(m.category))
        .sort((a, b) => b.qualityScore - a.qualityScore)
        .slice(0, config.maxOutputPerAgent);

      const noise = scored.filter((m) => m.qualityScore < config.minQualityScore || m.category === 'noise');
      const preFilteredIds = memories.filter((m) => !filtered.find((f) => f.id === m.id)).map((m) => m.id);

      selectedByAgent[agent] = gold;
      noiseByAgent[agent] = [...noise.map((m) => m.id), ...preFilteredIds];

      report.totalProcessed += memories.length;
      report.totalKept += gold.length;
      report.byAgent[agent] = {
        processed: memories.length, kept: gold.length,
        noiseCount: noiseByAgent[agent].length,
        topMemories: gold.slice(0, 5).map((m) => ({
          text: truncate(m.text, 160), score: m.qualityScore, category: m.category,
        })),
      };
    }

    report.totalDiscarded = report.totalProcessed - report.totalKept;
    const flatGold = Object.values(selectedByAgent).flat();
    const flatNoise = Object.values(noiseByAgent).flat();

    if (!config.dryRun) {
      // Step 1: Mark gold as distilled=true in mrc_bot_memory (PROTECTED)
      const marked = await this.qdrantService.markAsDistilled(flatGold);
      report.totalMarkedDistilled = marked;
      console.log(`\n✅ Marked ${marked} gold as distilled (protected forever)`);

      // Step 2: Delete noise from mrc_bot_memory
      const deleted = await this.qdrantService.deleteNoiseMemories(flatNoise);
      report.totalNoiseDeleted = deleted;
      console.log(`🗑️  Deleted ${deleted} noise from source`);

      // Step 3: Backup gold to golden collection
      await this.qdrantService.createGoldenCollection();
      await this.qdrantService.upsertGoldenMemories(flatGold);
      console.log(`📦 Backed up ${flatGold.length} to golden collection`);

      // Step 4: Save state for next incremental run
      const afterCount = await this.qdrantService.getCount(this.qdrantService.sourceCollectionName);
      const totalDistilled = await this.qdrantService.getDistilledCount();

      const newState: DistillState = {
        lastDistillTime: Date.now(),
        lastDistillDate: new Date().toISOString(),
        goldCount: flatGold.length,
        noiseDeleted: deleted,
        sourceCountBefore: beforeCount,
        sourceCountAfter: afterCount,
        totalDistilledEver: totalDistilled,
      };
      await this.qdrantService.saveState(newState);
      console.log(`💾 State saved to last-distill.json`);

      console.log(`\n📊 Result: ${beforeCount} → ${afterCount} points (cleaned ${beforeCount - afterCount})`);
      console.log(`📊 Protected (distilled=true): ${totalDistilled}`);

      // Step 5: Optional snapshot
      if (config.createSnapshot) {
        report.snapshotCreated = await this.qdrantService.createSnapshot(this.qdrantService.sourceCollectionName);
      }
    } else {
      console.log(`\n🔍 DRY RUN — no changes made`);
      console.log(`   Would keep: ${flatGold.length} gold, delete: ${flatNoise.length} noise`);
    }

    return report;
  }

  async buildReport(agents: string[]): Promise<DistillReport> {
    const categories: MemoryCategory[] = [
      'trading_win_pattern', 'trading_loss_lesson', 'market_insight',
      'bug_fix_pattern', 'architecture_decision', 'code_pattern',
      'process_improvement', 'project_context', 'system_rule',
    ];
    return this.distill({
      agents, minQualityScore: 60, maxOutputPerAgent: 5,
      categories, dryRun: true, createSnapshot: false, forceRuleOnly: true,
    });
  }
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 3)}...` : text;
}
