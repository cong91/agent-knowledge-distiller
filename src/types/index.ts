export interface AgentMemory {
  id: string;
  text: string;
  namespace: string;
  source_agent: string;
  source_type: string;
  userId: string;
  timestamp: number;
  vector?: number[];
}

export interface ScoredMemory extends AgentMemory {
  qualityScore: number; // 0-100
  category: MemoryCategory;
  tags: string[];
  distilledText?: string; // LLM-summarized version
  scoringMethod: 'rule' | 'llm';
  llmReasoning?: string;
}

export type MemoryCategory =
  | 'trading_win_pattern'
  | 'trading_loss_lesson'
  | 'market_insight'
  | 'bug_fix_pattern'
  | 'architecture_decision'
  | 'code_pattern'
  | 'process_improvement'
  | 'project_context'
  | 'system_rule'
  | 'noise';

export interface DistillConfig {
  agents: string[];
  minQualityScore: number;
  maxOutputPerAgent: number;
  categories: MemoryCategory[];
  dryRun?: boolean;
  createSnapshot?: boolean;
  forceRuleOnly?: boolean;
}

export interface DistillReport {
  timestamp: string;
  totalProcessed: number;
  totalKept: number;
  totalDiscarded: number;
  byAgent: Record<
    string,
    {
      processed: number;
      kept: number;
      topMemories: Array<{ text: string; score: number; category: string }>;
    }
  >;
  snapshotCreated?: string;
}
