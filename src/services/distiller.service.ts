import { DistillConfig, DistillReport, MemoryCategory, ScoredMemory } from '../types';
import { QdrantService } from './qdrant.service';
import { LLMScorerService } from './llm-scorer.service';
import { preFilter, scoreMemory } from './scorer.service';

export class DistillerService {
  constructor(private readonly qdrantService: QdrantService) {}

  /**
   * FULL distill: score ALL memories, mark gold, delete noise
   * Use for: first run, or when you want to re-evaluate everything
   */
  async distill(config: DistillConfig): Promise<DistillReport> {
    return this.runDistill(config, 'full');
  }

  /**
   * INCREMENTAL distill: score only NEW (undistilled) memories
   * Use for: daily runs — fast, cheap, keeps collection clean
   * 
   * Flow:
   * 1. Get memories WHERE distilled != true
   * 2. Score them (LLM or rule-based)
   * 3. Gold → mark distilled=true in mrc_bot_memory (PROTECTED forever)
   * 4. Noise → DELETE from mrc_bot_memory
   * 5. Gold → also upsert to agent_golden_knowledge (backup)
   */
  async incrementalDistill(config: DistillConfig): Promise<DistillReport> {
    return this.runDistill(config, 'incremental');
  }

  private async runDistill(
    config: DistillConfig, 
    mode: 'full' | 'incremental'
  ): Promise<DistillReport> {
    const report: DistillReport = {
      timestamp: new Date().toISOString(),
      mode,
      totalProcessed: 0,
      totalKept: 0,
      totalDiscarded: 0,
      totalNoiseDeleted: 0,
      totalMarkedDistilled: 0,
      byAgent: {},
    };

    const selectedByAgent: Record<string, ScoredMemory[]> = {};
    const noiseByAgent: Record<string, string[]> = {};

    // LLM config
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
      console.log('📏 Rule-based scoring (set LLM_SCORING_ENABLED=true for LLM)');
    }

    console.log(`📋 Mode: ${mode.toUpperCase()}`);

    // Get before counts
    const beforeCount = await this.qdrantService.getCount(this.qdrantService.sourceCollectionName);
    console.log(`📊 Source collection: ${beforeCount} points`);

    for (const agent of config.agents) {
      // INCREMENTAL: only undistilled memories | FULL: all memories
      const memories = mode === 'incremental'
        ? await this.qdrantService.getUndistilledMemories(agent)
        : await this.qdrantService.getAllMemories(agent);

      if (memories.length === 0) {
        console.log(`  [${agent}] No ${mode === 'incremental' ? 'new' : ''} memories to process`);
        continue;
      }

      const filtered = memories.filter(preFilter);
      console.log(
        `  [${agent}] ${mode === 'incremental' ? 'New' : 'All'}: ${memories.length} → pre-filter: ${filtered.length}`,
      );

      // Score
      const scored = llmScorer 
        ? await llmScorer.scoreBatch(filtered) 
        : filtered.map((memory) => scoreMemory(memory));

      // Split: gold vs noise
      const gold = scored
        .filter((m) => m.qualityScore >= config.minQualityScore)
        .filter((m) => m.category !== 'noise')
        .filter((m) => config.categories.includes(m.category))
        .sort((a, b) => b.qualityScore - a.qualityScore)
        .slice(0, config.maxOutputPerAgent);

      const noise = scored.filter(
        (m) => m.qualityScore < config.minQualityScore || m.category === 'noise'
      );

      // Also include pre-filtered (removed by preFilter) as noise
      const preFilteredIds = memories
        .filter((m) => !filtered.find((f) => f.id === m.id))
        .map((m) => m.id);

      selectedByAgent[agent] = gold;
      noiseByAgent[agent] = [...noise.map((m) => m.id), ...preFilteredIds];

      report.totalProcessed += memories.length;
      report.totalKept += gold.length;
      report.byAgent[agent] = {
        processed: memories.length,
        kept: gold.length,
        noiseCount: noiseByAgent[agent].length,
        topMemories: gold.slice(0, 5).map((memory) => ({
          text: truncate(memory.text, 160),
          score: memory.qualityScore,
          category: memory.category,
        })),
      };
    }

    report.totalDiscarded = report.totalProcessed - report.totalKept;

    const flatGold = Object.values(selectedByAgent).flat();
    const flatNoise = Object.values(noiseByAgent).flat();

    if (!config.dryRun) {
      // 1. Mark gold as distilled IN-PLACE in mrc_bot_memory
      const marked = await this.qdrantService.markAsDistilled(flatGold);
      report.totalMarkedDistilled = marked;
      console.log(`\n✅ Marked ${marked} gold memories as distilled (protected forever)`);

      // 2. Delete noise from mrc_bot_memory
      const deleted = await this.qdrantService.deleteNoiseMemories(flatNoise);
      report.totalNoiseDeleted = deleted;
      console.log(`🗑️  Deleted ${deleted} noise memories from source`);

      // 3. Also upsert gold to golden collection (backup/reference)
      await this.qdrantService.createGoldenCollection();
      await this.qdrantService.upsertGoldenMemories(flatGold);
      console.log(`📦 Upserted ${flatGold.length} to golden collection (backup)`);

      // 4. After counts
      const afterCount = await this.qdrantService.getCount(this.qdrantService.sourceCollectionName);
      console.log(`\n📊 Source collection: ${beforeCount} → ${afterCount} points (cleaned ${beforeCount - afterCount})`);

      if (config.createSnapshot) {
        report.snapshotCreated = await this.qdrantService.createSnapshot(
          this.qdrantService.sourceCollectionName
        );
      }
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
