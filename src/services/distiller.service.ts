import { DistillConfig, DistillReport, MemoryCategory, ScoredMemory } from '../types';
import { QdrantService } from './qdrant.service';
import { LLMScorerService } from './llm-scorer.service';
import { preFilter, scoreMemory } from './scorer.service';

export class DistillerService {
  constructor(private readonly qdrantService: QdrantService) {}

  async distill(config: DistillConfig): Promise<DistillReport> {
    const report: DistillReport = {
      timestamp: new Date().toISOString(),
      totalProcessed: 0,
      totalKept: 0,
      totalDiscarded: 0,
      byAgent: {},
    };

    const selectedByAgent: Record<string, ScoredMemory[]> = {};

    // OpenAI-compatible LLM config (same pattern as agent-smart-memo)
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

    for (const agent of config.agents) {
      const memories = await this.qdrantService.getAllMemories(agent);

      const filtered = memories.filter(preFilter);
      console.log(
        `  [${agent}] Pre-filter: ${memories.length} → ${filtered.length} (removed ${memories.length - filtered.length} noise)`,
      );

      const scored = llmScorer ? await llmScorer.scoreBatch(filtered) : filtered.map((memory) => scoreMemory(memory));

      const top = scored
        .filter((memory) => memory.qualityScore >= config.minQualityScore)
        .filter((memory) => memory.category !== 'noise')
        .filter((memory) => config.categories.includes(memory.category))
        .sort((a, b) => b.qualityScore - a.qualityScore)
        .slice(0, config.maxOutputPerAgent);

      selectedByAgent[agent] = top;

      report.totalProcessed += filtered.length;
      report.totalKept += top.length;
      report.byAgent[agent] = {
        processed: filtered.length,
        kept: top.length,
        topMemories: top.slice(0, 5).map((memory) => ({
          text: truncate(memory.text, 160),
          score: memory.qualityScore,
          category: memory.category,
        })),
      };
    }

    report.totalDiscarded = report.totalProcessed - report.totalKept;

    const flatMemories = Object.values(selectedByAgent).flat();

    if (!config.dryRun) {
      await this.qdrantService.createGoldenCollection();
      await this.qdrantService.upsertGoldenMemories(flatMemories);

      if (config.createSnapshot) {
        report.snapshotCreated = await this.qdrantService.createSnapshot(this.qdrantService.goldenCollectionName);
      }
    }

    return report;
  }

  async buildReport(agents: string[]): Promise<DistillReport> {
    const categories: MemoryCategory[] = [
      'trading_win_pattern',
      'trading_loss_lesson',
      'market_insight',
      'bug_fix_pattern',
      'architecture_decision',
      'code_pattern',
      'process_improvement',
      'project_context',
      'system_rule',
    ];

    return this.distill({
      agents,
      minQualityScore: 60,
      maxOutputPerAgent: 5,
      categories,
      dryRun: true,
      createSnapshot: false,
      forceRuleOnly: true,
    });
  }
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 3)}...` : text;
}
