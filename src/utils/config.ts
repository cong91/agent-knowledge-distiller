import dotenv from 'dotenv';
import { DistillConfig, MemoryCategory } from '../types';

dotenv.config();

export const DEFAULT_AGENTS = ['trader', 'fullstack', 'assistant', 'scrum'];

export const DEFAULT_CATEGORIES: MemoryCategory[] = [
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

export function buildDistillConfig(options: {
  agents?: string[];
  minScore?: number;
  maxPerAgent?: number;
  dryRun?: boolean;
  createSnapshot?: boolean;
}): DistillConfig {
  return {
    agents: options.agents && options.agents.length > 0 ? options.agents : DEFAULT_AGENTS,
    minQualityScore: options.minScore ?? 60,
    maxOutputPerAgent: options.maxPerAgent ?? 100,
    categories: DEFAULT_CATEGORIES,
    dryRun: options.dryRun ?? false,
    createSnapshot: options.createSnapshot ?? false,
  };
}
