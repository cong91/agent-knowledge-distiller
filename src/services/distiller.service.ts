import { DistillConfig, DistillReport, MemoryCategory, ScoredMemory } from '../types';
import { QdrantService } from './qdrant.service';
import { ScorerService } from './scorer.service';

export class DistillerService {
  constructor(
    private readonly qdrantService: QdrantService,
    private readonly scorerService: ScorerService,
  ) {}

  async distill(config: DistillConfig): Promise<DistillReport> {
    const report: DistillReport = {
      timestamp: new Date().toISOString(),
      totalProcessed: 0,
      totalKept: 0,
      totalDiscarded: 0,
      byAgent: {},
    };

    const selectedByAgent: Record<string, ScoredMemory[]> = {};

    for (const agent of config.agents) {
      const memories = await this.qdrantService.getAllMemories(agent);
      const scored = memories.map((memory) => this.scorerService.scoreMemory(memory));

      const filtered = scored
        .filter((memory) => memory.qualityScore >= config.minQualityScore)
        .filter((memory) => memory.category !== 'noise')
        .filter((memory) => config.categories.includes(memory.category));

      const top = filtered
        .sort((a, b) => b.qualityScore - a.qualityScore)
        .slice(0, config.maxOutputPerAgent);

      selectedByAgent[agent] = top;

      report.totalProcessed += scored.length;
      report.totalKept += top.length;
      report.byAgent[agent] = {
        processed: scored.length,
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
        report.snapshotCreated = await this.qdrantService.createSnapshot(
          this.qdrantService.goldenCollectionName,
        );
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
    });
  }
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 3)}...` : text;
}
