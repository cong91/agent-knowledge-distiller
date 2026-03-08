import { DistillConfig, DistillReport, DistillState, MemoryCategory, ScoredMemory } from '../types';
import { QdrantService } from './qdrant.service';
import { LLMScorerService } from './llm-scorer.service';
import { preFilter, scoreMemory } from './scorer.service';

export class DistillerService {
  constructor(private readonly qdrantService: QdrantService) {}

  async distill(config: DistillConfig): Promise<DistillReport> {
    return this.runDistill(config, 'full');
  }

  async incrementalDistill(config: DistillConfig): Promise<DistillReport> {
    return this.runDistill(config, 'incremental');
  }

  private async runDistill(config: DistillConfig, mode: 'full' | 'incremental'): Promise<DistillReport> {
    const report: DistillReport = {
      timestamp: new Date().toISOString(), mode,
      totalProcessed: 0, totalKept: 0, totalDiscarded: 0,
      totalNoiseDeleted: 0, totalMarkedDistilled: 0, totalReembedded: 0,
      byAgent: {},
    };

    const selectedByAgent: Record<string, ScoredMemory[]> = {};
    const noiseByAgent: Record<string, string[]> = {};

    // LLM setup
    const llmBaseUrl = process.env.LLM_BASE_URL || 'http://localhost:8317/v1';
    const llmApiKey = process.env.LLM_API_KEY || 'proxypal-local';
    const llmModel = process.env.LLM_MODEL || 'gpt-5';
    const llmEnabled = process.env.LLM_SCORING_ENABLED === 'true' && !config.forceRuleOnly;

    let llmScorer: LLMScorerService | null = null;
    if (llmEnabled) {
      llmScorer = new LLMScorerService(
        { baseUrl: llmBaseUrl, apiKey: llmApiKey, model: llmModel },
        Number.parseInt(process.env.LLM_BATCH_SIZE || '10', 10),
      );
      console.log(`🧠 LLM scoring + enriching enabled (${llmModel})`);
    } else {
      console.log('📏 Rule-based scoring (no enrichment — set --llm for LLM)');
    }

    const prevState = await this.qdrantService.loadState();
    if (mode === 'incremental' && prevState) {
      console.log(`📋 Mode: INCREMENTAL (since ${prevState.lastDistillDate})`);
    } else {
      console.log(`📋 Mode: ${mode.toUpperCase()}`);
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

      // Gold = score >= threshold AND not noise (no arbitrary limit)
      const gold = scored
        .filter((m) => m.qualityScore >= config.minQualityScore)
        .filter((m) => m.category !== 'noise')
        .sort((a, b) => b.qualityScore - a.qualityScore);

      // Everything else = noise (to be deleted)
      const goldIds = new Set(gold.map((m) => m.id));
      const noise = scored.filter((m) => !goldIds.has(m.id));
      const preFilteredIds = memories.filter((m) => !filtered.find((f) => f.id === m.id)).map((m) => m.id);

      selectedByAgent[agent] = gold;
      noiseByAgent[agent] = [...noise.map((m) => m.id), ...preFilteredIds];

      report.totalProcessed += memories.length;
      report.totalKept += gold.length;
      report.byAgent[agent] = {
        processed: memories.length, kept: gold.length,
        noiseCount: noiseByAgent[agent].length,
        topMemories: gold.slice(0, 3).map((m) => ({
          text: truncate(m.text, 120), score: m.qualityScore, category: m.category,
          enrichedText: truncate(m.distilledText || '', 150),
        })),
      };
    }

    report.totalDiscarded = report.totalProcessed - report.totalKept;
    const flatGold = Object.values(selectedByAgent).flat();
    const flatNoise = Object.values(noiseByAgent).flat();

    if (!config.dryRun) {
      // Step 1: Mark gold as distilled + RE-EMBED enrichedText
      console.log(`\n🔄 Enriching + re-embedding ${flatGold.length} gold memories...`);
      const { marked, reembedded, truncated, embeddingMeta } = await this.qdrantService.markAsDistilledAndReembed(flatGold);
      report.totalMarkedDistilled = marked;
      report.totalReembedded = reembedded;
      report.totalTruncated = truncated;
      console.log(`✅ Marked: ${marked}, Re-embedded: ${reembedded}, Truncated: ${truncated}`);

      // Update topMemories with embedding metadata
      for (const [agent, agentData] of Object.entries(report.byAgent)) {
        agentData.topMemories = agentData.topMemories.map((m, idx) => {
          const originalMemory = selectedByAgent[agent]?.[idx];
          if (originalMemory && embeddingMeta.has(originalMemory.id)) {
            return { ...m, embeddingMetadata: embeddingMeta.get(originalMemory.id) };
          }
          return m;
        });
      }

      // Step 2: Delete noise
      const deleted = await this.qdrantService.deleteNoiseMemories(flatNoise);
      report.totalNoiseDeleted = deleted;
      console.log(`🗑️  Deleted ${deleted} noise`);

      // Step 3: Backup to golden collection
      await this.qdrantService.createGoldenCollection();
      await this.qdrantService.upsertGoldenMemories(flatGold);
      console.log(`📦 Backed up ${flatGold.length} to golden collection`);

      // Step 4: Save state
      const afterCount = await this.qdrantService.getCount(this.qdrantService.sourceCollectionName);
      const totalDistilled = await this.qdrantService.getDistilledCount();
      await this.qdrantService.saveState({
        lastDistillTime: Date.now(),
        lastDistillDate: new Date().toISOString(),
        goldCount: flatGold.length,
        noiseDeleted: deleted,
        sourceCountBefore: beforeCount,
        sourceCountAfter: afterCount,
        totalDistilledEver: totalDistilled,
      });

      console.log(`\n📊 Source: ${beforeCount} → ${afterCount} (cleaned ${beforeCount - afterCount})`);
      console.log(`📊 Protected (distilled=true): ${totalDistilled}`);
      console.log(`📊 Re-embedded vectors: ${reembedded}`);

      if (config.createSnapshot) {
        report.snapshotCreated = await this.qdrantService.createSnapshot(this.qdrantService.sourceCollectionName);
      }
    } else {
      console.log(`\n🔍 DRY RUN — no changes`);
      console.log(`   Would: keep ${flatGold.length} gold (enrich+re-embed), delete ${flatNoise.length} noise`);
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
