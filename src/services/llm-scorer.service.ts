import { AgentMemory, MemoryCategory, ScoredMemory } from '../types';
import { scoreMemory } from './scorer.service';

/**
 * LLM Context-Aware Scorer Service
 * 
 * Improvement: Instead of scoring each memory in isolation,
 * groups memories by agent + time window (±30min) so LLM
 * sees surrounding context when evaluating each memory.
 * 
 * Example: "Tập trung quản lý 3 vị thế: KITE, SIGN, POWER"
 * → alone = low score (vague)
 * → with context of trade journal entries nearby = high score
 */

interface LLMConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

interface OpenAIChatResponse {
  choices: Array<{
    message: { content: string; role: string };
    finish_reason: string;
    index: number;
  }>;
  model: string;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

// 30 minutes in ms — memories within this window are "related context"
const CONTEXT_WINDOW_MS = 30 * 60 * 1000;

const SCORING_PROMPT = `You are an AI Knowledge Quality Evaluator. Score each memory for its long-term value to an AI agent team.

Context: These memories come from an AI trading system with 4 agents:
- trader: Analyzes crypto markets, executes BUY/SELL/HOLD decisions
- fullstack: Builds backend (NestJS/TypeScript), fixes bugs, implements features
- scrum: Project management, planning, orchestration
- assistant: General orchestration, user interaction

IMPORTANT: Each memory marked [SCORE] needs a score. Memories marked [CTX] are surrounding context — DO NOT score them, they are provided so you understand what was happening around the memory being scored.

For each [SCORE] memory, provide:
1. score (0-100): How valuable is this knowledge for the agent's future performance?
2. category: One of: trading_win_pattern, trading_loss_lesson, market_insight, bug_fix_pattern, architecture_decision, code_pattern, process_improvement, project_context, system_rule, noise
3. reasoning: 1 sentence why this score

Scoring guidelines:
- 90-100: Critical lesson that prevents real money loss or saves days of work
- 70-89: Valuable pattern/insight that improves decision making
- 50-69: Useful context but not directly actionable
- 30-49: Low value, temporary or already outdated info
- 0-29: Noise, test data, system junk, or duplicated info

Respond in JSON array format (ONLY for [SCORE] memories):
[{"index": 0, "score": 85, "category": "trading_win_pattern", "reasoning": "..."}]`;

const ALLOWED_CATEGORIES = new Set<MemoryCategory>([
  'trading_win_pattern', 'trading_loss_lesson', 'market_insight',
  'bug_fix_pattern', 'architecture_decision', 'code_pattern',
  'process_improvement', 'project_context', 'system_rule', 'noise',
]);

export class LLMScorerService {
  private config: LLMConfig;
  private batchSize: number;
  private allMemoriesIndex: Map<string, AgentMemory[]>; // agent → sorted memories

  constructor(config: LLMConfig, batchSize = 10) {
    this.config = config;
    this.batchSize = Math.max(1, batchSize);
    this.allMemoriesIndex = new Map();
  }

  /**
   * Score with context: builds an index of ALL memories first,
   * then for each batch, includes nearby context memories
   */
  async scoreBatch(memories: AgentMemory[]): Promise<ScoredMemory[]> {
    // Build time-sorted index per agent for context lookup
    this.buildContextIndex(memories);

    const results: ScoredMemory[] = [];

    for (let i = 0; i < memories.length; i += this.batchSize) {
      const batch = memories.slice(i, i + this.batchSize);
      const batchResults = await this.scoreSingleBatch(batch);
      results.push(...batchResults);

      const progress = Math.min(i + this.batchSize, memories.length);
      console.log(`  🧠 LLM scoring: ${progress}/${memories.length} (${((progress / memories.length) * 100).toFixed(0)}%)`);

      if (i + this.batchSize < memories.length) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    return results;
  }

  private buildContextIndex(memories: AgentMemory[]): void {
    this.allMemoriesIndex.clear();
    for (const m of memories) {
      const agent = m.source_agent || 'unknown';
      if (!this.allMemoriesIndex.has(agent)) {
        this.allMemoriesIndex.set(agent, []);
      }
      this.allMemoriesIndex.get(agent)!.push(m);
    }
    // Sort each agent's memories by timestamp
    for (const [, list] of this.allMemoriesIndex) {
      list.sort((a, b) => a.timestamp - b.timestamp);
    }
  }

  /**
   * Find surrounding memories within ±30min of target memory
   * Returns up to 3 context memories (before + after)
   */
  private findContextMemories(target: AgentMemory): AgentMemory[] {
    const agentMemories = this.allMemoriesIndex.get(target.source_agent || 'unknown') || [];
    const context: AgentMemory[] = [];

    for (const m of agentMemories) {
      if (m.id === target.id) continue; // skip self
      const timeDiff = Math.abs(m.timestamp - target.timestamp);
      if (timeDiff <= CONTEXT_WINDOW_MS) {
        context.push(m);
      }
    }

    // Return max 3 closest context memories (to limit prompt size)
    return context
      .sort((a, b) => Math.abs(a.timestamp - target.timestamp) - Math.abs(b.timestamp - target.timestamp))
      .slice(0, 3);
  }

  private async scoreSingleBatch(batch: AgentMemory[]): Promise<ScoredMemory[]> {
    // Build context-enriched prompt
    const memoriesText = batch.map((m, idx) => {
      const contextMemories = this.findContextMemories(m);
      
      let text = `[SCORE ${idx}] agent=${m.source_agent} ns=${m.namespace} | ${(m.text ?? '').slice(0, 500)}`;
      
      if (contextMemories.length > 0) {
        const ctxLines = contextMemories.map(
          (c) => `  [CTX] ${(c.text ?? '').slice(0, 200)}`
        ).join('\n');
        text += `\n  ↳ Surrounding context (same agent, ±30min):\n${ctxLines}`;
      }
      
      return text;
    }).join('\n\n');

    try {
      const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.model,
          messages: [
            { role: 'system', content: SCORING_PROMPT },
            { role: 'user', content: `Memories to score:\n\n${memoriesText}` },
          ],
          temperature: 0.1,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`LLM API error: ${response.status} - ${errorText.slice(0, 200)}`);
      }

      const data = await response.json() as OpenAIChatResponse;
      const responseText = data.choices?.[0]?.message?.content;

      if (!responseText) {
        throw new Error('No content in LLM response');
      }

      const scores = this.parseScores(responseText);

      return batch.map((memory, idx) => {
        const llmScore = scores.find((s) => s.index === idx);
        if (!llmScore) {
          const fallback = scoreMemory(memory);
          return { ...fallback, tags: [...fallback.tags, 'llm-fallback'] };
        }

        const category: MemoryCategory = ALLOWED_CATEGORIES.has(llmScore.category) 
          ? llmScore.category : 'noise';

        return {
          ...memory,
          qualityScore: clampScore(llmScore.score),
          category,
          tags: [category, 'llm-scored', 'context-aware'],
          distilledText: llmScore.reasoning,
          scoringMethod: 'llm' as const,
          llmReasoning: llmScore.reasoning,
        };
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`  ⚠️ LLM batch failed, falling back to rule-based: ${message}`);
      return batch.map((memory) => {
        const fallback = scoreMemory(memory);
        return { ...fallback, tags: [...fallback.tags, 'llm-fallback'] };
      });
    }
  }

  private parseScores(rawText: string): Array<{
    index: number; score: number; category: MemoryCategory; reasoning: string;
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
