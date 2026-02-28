import { GoogleGenerativeAI } from '@google/generative-ai';
import { AgentMemory, MemoryCategory, ScoredMemory } from '../types';
import { scoreMemory } from './scorer.service';

const SCORING_PROMPT = `You are an AI Knowledge Quality Evaluator. Score each memory for its long-term value to an AI agent team.

Context: These memories come from an AI trading system with 4 agents:
- trader: Analyzes crypto markets, executes BUY/SELL/HOLD decisions
- fullstack: Builds backend (NestJS/TypeScript), fixes bugs, implements features
- scrum: Project management, planning, orchestration
- assistant: General orchestration, user interaction

For each memory, provide:
1. score (0-100): How valuable is this knowledge for the agent's future performance?
2. category: One of: trading_win_pattern, trading_loss_lesson, market_insight, bug_fix_pattern, architecture_decision, code_pattern, process_improvement, project_context, system_rule, noise
3. reasoning: 1 sentence why this score

Scoring guidelines:
- 90-100: Critical lesson that prevents real money loss or saves days of work
- 70-89: Valuable pattern/insight that improves decision making
- 50-69: Useful context but not directly actionable
- 30-49: Low value, temporary or already outdated info
- 0-29: Noise, test data, system junk, or duplicated info

Respond in JSON array format:
[{"index": 0, "score": 85, "category": "trading_win_pattern", "reasoning": "..."}]`;

const ALLOWED_CATEGORIES = new Set<MemoryCategory>([
  'trading_win_pattern',
  'trading_loss_lesson',
  'market_insight',
  'bug_fix_pattern',
  'architecture_decision',
  'code_pattern',
  'process_improvement',
  'project_context',
  'system_rule',
  'noise',
]);

export class LLMScorerService {
  private genAI: GoogleGenerativeAI;
  private model: string;
  private batchSize: number;

  constructor(apiKey: string, model = 'gemini-2.5-flash', batchSize = 10) {
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = model;
    this.batchSize = Math.max(1, batchSize);
  }

  async scoreBatch(memories: AgentMemory[]): Promise<ScoredMemory[]> {
    const results: ScoredMemory[] = [];

    for (let i = 0; i < memories.length; i += this.batchSize) {
      const batch = memories.slice(i, i + this.batchSize);
      const batchResults = await this.scoreSingleBatch(batch);
      results.push(...batchResults);

      const progress = Math.min(i + this.batchSize, memories.length);
      console.log(`  LLM scoring: ${progress}/${memories.length} (${((progress / memories.length) * 100).toFixed(0)}%)`);

      if (i + this.batchSize < memories.length) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    return results;
  }

  private async scoreSingleBatch(batch: AgentMemory[]): Promise<ScoredMemory[]> {
    const model = this.genAI.getGenerativeModel({ model: this.model });

    const memoriesText = batch
      .map((m, idx) => `[${idx}] agent=${m.source_agent} ns=${m.namespace} | ${(m.text ?? '').slice(0, 500)}`)
      .join('\n\n');

    try {
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: `${SCORING_PROMPT}\n\nMemories to score:\n\n${memoriesText}` }] }],
        generationConfig: {
          temperature: 0.1,
          responseMimeType: 'application/json',
        },
      });

      const responseText = result.response.text();
      const scores = this.parseScores(responseText);

      return batch.map((memory, idx) => {
        const llmScore = scores.find((s) => s.index === idx);
        if (!llmScore) {
          const fallback = scoreMemory(memory);
          return {
            ...fallback,
            tags: [...fallback.tags, 'llm-fallback'],
          };
        }

        const category: MemoryCategory = ALLOWED_CATEGORIES.has(llmScore.category) ? llmScore.category : 'noise';

        return {
          ...memory,
          qualityScore: clampScore(llmScore.score),
          category,
          tags: [category, 'llm-scored'],
          distilledText: llmScore.reasoning,
          scoringMethod: 'llm',
          llmReasoning: llmScore.reasoning,
        };
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`  LLM batch scoring failed, falling back to rule-based: ${message}`);
      return batch.map((memory) => {
        const fallback = scoreMemory(memory);
        return {
          ...fallback,
          tags: [...fallback.tags, 'llm-fallback'],
        };
      });
    }
  }

  private parseScores(rawText: string): Array<{
    index: number;
    score: number;
    category: MemoryCategory;
    reasoning: string;
  }> {
    const cleaned = stripCodeFence(rawText).trim();
    const parsed = JSON.parse(cleaned) as unknown;

    if (!Array.isArray(parsed)) {
      throw new Error('LLM response is not a JSON array');
    }

    return parsed
      .map((item) => {
        const obj = item as Record<string, unknown>;
        return {
          index: Number(obj.index),
          score: Number(obj.score),
          category: String(obj.category ?? 'noise') as MemoryCategory,
          reasoning: String(obj.reasoning ?? 'LLM scoring'),
        };
      })
      .filter((item) => Number.isFinite(item.index) && Number.isFinite(item.score));
  }
}

function stripCodeFence(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced?.[1]) return fenced[1];
  return text;
}

function clampScore(score: number): number {
  if (!Number.isFinite(score)) return 50;
  return Math.min(100, Math.max(0, Math.round(score)));
}
