import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { QdrantClient } from '@qdrant/js-client-rest';
import { AgentMemory, ScoredMemory, DistillState } from '../types';

const DEFAULT_BATCH_SIZE = 256;

type ScrollOffset = string | number | Record<string, unknown> | null | undefined;

export class QdrantService {
  private readonly client: QdrantClient;
  private readonly sourceCollection: string;
  private readonly goldenCollection: string;
  private readonly snapshotDir: string;
  private readonly stateFile: string;

  constructor() {
    const host = process.env.QDRANT_HOST ?? 'localhost';
    const port = Number(process.env.QDRANT_PORT ?? 6333);
    this.client = new QdrantClient({ host, port });
    this.sourceCollection = process.env.QDRANT_COLLECTION ?? 'mrc_bot_memory';
    this.goldenCollection = process.env.GOLDEN_COLLECTION ?? 'agent_golden_knowledge';
    this.snapshotDir = process.env.SNAPSHOT_DIR ?? './snapshots';
    this.stateFile = path.join(this.snapshotDir, 'last-distill.json');
  }

  get sourceCollectionName(): string { return this.sourceCollection; }
  get goldenCollectionName(): string { return this.goldenCollection; }

  // =================== MEMORY RETRIEVAL ===================

  /** FULL: get ALL memories */
  async getAllMemories(agent?: string, namespace?: string): Promise<AgentMemory[]> {
    return this.scrollMemories(this.sourceCollection, agent, namespace);
  }

  /**
   * INCREMENTAL: get only NEW undistilled memories (2-layer protection)
   * Layer 1: skip distilled=true (flag on record)
   * Layer 2: only get timestamp > lastDistillTime (from state file)
   */
  async getUndistilledMemories(agent?: string, namespace?: string): Promise<AgentMemory[]> {
    const state = await this.loadState();
    return this.scrollMemories(this.sourceCollection, agent, namespace, {
      skipDistilled: true,
      sinceTimestamp: state?.lastDistillTime,
    });
  }

  // =================== DISTILL ACTIONS ===================

  /**
   * Mark gold memories as distilled IN-PLACE in mrc_bot_memory
   * Sets: distilled=true, distilledAt, qualityScore, category
   * PROTECTED: never deleted, never re-scored
   */
  async markAsDistilled(memories: ScoredMemory[]): Promise<number> {
    if (memories.length === 0) return 0;
    const now = Date.now();
    let marked = 0;
    for (const memory of memories) {
      try {
        await this.client.setPayload(this.sourceCollection, {
          points: [memory.id],
          payload: {
            distilled: true, distilledAt: now,
            qualityScore: memory.qualityScore, category: memory.category,
            tags: memory.tags, scoringMethod: memory.scoringMethod,
            llmReasoning: memory.llmReasoning, distilledText: memory.distilledText,
          },
        });
        marked++;
      } catch (err) {
        console.error(`  ⚠️ Failed to mark ${memory.id}: ${err}`);
      }
    }
    return marked;
  }

  /** Delete noise memories from mrc_bot_memory */
  async deleteNoiseMemories(noiseIds: string[]): Promise<number> {
    if (noiseIds.length === 0) return 0;
    const chunkSize = 128;
    let deleted = 0;
    for (let i = 0; i < noiseIds.length; i += chunkSize) {
      const chunk = noiseIds.slice(i, i + chunkSize);
      try {
        await this.client.delete(this.sourceCollection, { wait: true, points: chunk });
        deleted += chunk.length;
      } catch (err) {
        console.error(`  ⚠️ Failed to delete chunk: ${err}`);
      }
    }
    return deleted;
  }

  // =================== STATE MANAGEMENT ===================

  /** Load last distill state from snapshots/last-distill.json */
  async loadState(): Promise<DistillState | null> {
    try {
      const data = await readFile(this.stateFile, 'utf-8');
      return JSON.parse(data) as DistillState;
    } catch {
      return null; // First run — no state file yet
    }
  }

  /** Save distill state after successful run */
  async saveState(state: DistillState): Promise<void> {
    await mkdir(this.snapshotDir, { recursive: true });
    await writeFile(this.stateFile, JSON.stringify(state, null, 2));
  }

  // =================== GOLDEN COLLECTION ===================

  async createGoldenCollection(): Promise<void> {
    const exists = await this.client.collectionExists(this.goldenCollection);
    if (exists.exists) return;
    const sourceInfo = await this.client.getCollection(this.sourceCollection);
    const sourceVectors = sourceInfo.config?.params?.vectors;
    if (!sourceVectors) {
      await this.client.createCollection(this.goldenCollection, { vectors: { size: 1024, distance: 'Cosine' } });
      return;
    }
    await this.client.createCollection(this.goldenCollection, { vectors: sourceVectors });
  }

  async upsertGoldenMemories(memories: ScoredMemory[]): Promise<void> {
    if (memories.length === 0) return;
    const points = memories.map((m) => ({
      id: m.id,
      vector: m.vector ?? new Array(1024).fill(0),
      payload: {
        text: m.text, distilledText: m.distilledText, namespace: m.namespace,
        source_agent: m.source_agent, source_type: m.source_type, userId: m.userId,
        timestamp: m.timestamp, qualityScore: m.qualityScore, category: m.category,
        tags: m.tags, scoringMethod: m.scoringMethod, llmReasoning: m.llmReasoning,
      },
    }));
    const chunkSize = 128;
    for (let i = 0; i < points.length; i += chunkSize) {
      await this.client.upsert(this.goldenCollection, { wait: true, points: points.slice(i, i + chunkSize) });
    }
  }

  // =================== UTILS ===================

  async getCount(collectionName: string): Promise<number> {
    const info = await this.client.getCollection(collectionName);
    return info.points_count ?? 0;
  }

  async getDistilledCount(): Promise<number> {
    const result = await this.client.count(this.sourceCollection, {
      filter: { must: [{ key: 'distilled', match: { value: true } }] },
      exact: true,
    });
    return result.count;
  }

  async createSnapshot(collectionName: string): Promise<string> {
    const result = await this.client.createSnapshot(collectionName);
    if (!result?.name) throw new Error(`Failed to create snapshot for: ${collectionName}`);
    await mkdir(this.snapshotDir, { recursive: true });
    return path.join(this.snapshotDir, result.name);
  }

  async listSnapshots(collectionName: string): Promise<string[]> {
    const snapshots = await this.client.listSnapshots(collectionName);
    return snapshots.map((s) => s.name);
  }

  async getCollectionInfo(name: string) { return this.client.getCollection(name); }

  // =================== PRIVATE ===================

  private async scrollMemories(
    collectionName: string, agent?: string, namespace?: string,
    options?: { skipDistilled?: boolean; sinceTimestamp?: number },
  ): Promise<AgentMemory[]> {
    const memories: AgentMemory[] = [];
    let nextPageOffset: ScrollOffset = null;
    const filter = this.buildFilter(agent, namespace, options);

    do {
      const res = await this.client.scroll(collectionName, {
        limit: DEFAULT_BATCH_SIZE, offset: (nextPageOffset ?? undefined) as never,
        with_payload: true, with_vector: true, filter,
      });
      for (const point of res.points) {
        const payload = (point.payload ?? {}) as Record<string, unknown>;
        memories.push({
          id: String(point.id),
          text: typeof payload.text === 'string' ? payload.text : '',
          namespace: String(payload.namespace ?? ''), source_agent: String(payload.source_agent ?? ''),
          source_type: String(payload.source_type ?? ''), userId: String(payload.userId ?? ''),
          timestamp: Number(payload.timestamp ?? Date.now()),
          vector: Array.isArray(point.vector) ? (point.vector as number[]) : undefined,
        });
      }
      nextPageOffset = res.next_page_offset;
    } while (nextPageOffset !== null && nextPageOffset !== undefined);

    return memories;
  }

  private buildFilter(
    agent?: string, namespace?: string,
    options?: { skipDistilled?: boolean; sinceTimestamp?: number },
  ) {
    const must: Array<Record<string, unknown>> = [];
    const must_not: Array<Record<string, unknown>> = [];

    if (agent) must.push({ key: 'source_agent', match: { value: agent } });
    if (namespace) must.push({ key: 'namespace', match: { value: namespace } });

    // Layer 1: skip already-distilled records
    if (options?.skipDistilled) {
      must_not.push({ key: 'distilled', match: { value: true } });
    }

    // Layer 2: only memories newer than last distill time
    if (options?.sinceTimestamp) {
      must.push({ key: 'timestamp', range: { gt: options.sinceTimestamp } });
    }

    const filter: Record<string, unknown> = {};
    if (must.length > 0) filter.must = must;
    if (must_not.length > 0) filter.must_not = must_not;
    return Object.keys(filter).length > 0 ? filter : undefined;
  }
}
