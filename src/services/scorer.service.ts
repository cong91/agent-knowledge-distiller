import { AgentMemory, MemoryCategory, ScoredMemory } from '../types';

export class ScorerService {
  scoreMemory(memory: AgentMemory): ScoredMemory {
    return scoreMemory(memory);
  }
}

export function preFilter(memory: AgentMemory): boolean {
  const text = (memory.text ?? '').trim();
  const lower = text.toLowerCase();

  if (text.length < 20) return false;
  if (lower.includes('subagent direct hook test')) return false;
  if (lower.includes('skipping:')) return false;
  if (lower.includes('no output')) return false;
  if (/^(ok|yes|no|done|test)$/i.test(text)) return false;

  return true;
}

export function scoreMemory(memory: AgentMemory): ScoredMemory {
  let score = 50;
  const text = (memory.text ?? '').toLowerCase();
  const tags: string[] = [];

  if (text.includes('win') || text.includes('thắng') || text.includes('lợi nhuận')) {
    score += 15;
    tags.push('win');
  }
  if (text.includes('pattern') || text.includes('mẫu')) {
    score += 10;
    tags.push('pattern');
  }
  if (text.includes('fix') || text.includes('resolved') || text.includes('đã sửa')) {
    score += 10;
    tags.push('fix');
  }
  if (text.includes('rule') || text.includes('quy tắc') || text.includes('luật')) {
    score += 10;
    tags.push('rule');
  }
  if (text.includes('rsi') || text.includes('macd') || text.includes('sma')) {
    score += 5;
    tags.push('technical');
  }
  if (text.includes('architecture') || text.includes('design')) {
    score += 10;
    tags.push('architecture');
  }
  if (text.includes('lesson') || text.includes('bài học') || text.includes('kinh nghiệm')) {
    score += 15;
    tags.push('lesson');
  }
  if (text.length > 200) score += 5;
  if (text.length > 500) score += 5;

  if (text.includes('error') && text.includes('500')) score -= 10;
  if (text.length < 30) score -= 20;
  if (text.includes('skipping') || text.includes('no output')) score -= 15;
  if (text.includes('test') && !text.includes('backtest')) score -= 10;

  let category: MemoryCategory = 'noise';

  if (memory.source_agent === 'trader') {
    if (tags.includes('win')) category = 'trading_win_pattern';
    else if (text.includes('thua') || text.includes('loss') || text.includes('stop loss')) {
      category = 'trading_loss_lesson';
    } else if (tags.includes('technical') || text.includes('market') || text.includes('thị trường')) {
      category = 'market_insight';
    } else if (score > 50) {
      category = 'market_insight';
    }
  } else if (memory.source_agent === 'fullstack') {
    if (tags.includes('fix')) category = 'bug_fix_pattern';
    else if (tags.includes('architecture')) category = 'architecture_decision';
    else if (score > 50) category = 'code_pattern';
  } else if (memory.source_agent === 'scrum') {
    if (score > 50) category = 'process_improvement';
  } else if (memory.source_agent === 'assistant') {
    if (tags.includes('rule')) category = 'system_rule';
    else if (score > 55) category = 'project_context';
  }

  if (tags.includes('rule')) category = 'system_rule';
  if (score < 30) category = 'noise';

  return {
    ...memory,
    qualityScore: Math.min(100, Math.max(0, score)),
    category,
    tags,
    scoringMethod: 'rule',
  };
}
