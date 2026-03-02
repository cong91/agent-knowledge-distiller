import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { QdrantClient } from '@qdrant/js-client-rest';
import { AgentMemory, ScoredMemory, DistillState } from '../types';
import { EmbeddingService } from './embedding.service';

const DEFAULT_BATCH_SIZE = 256;
type ScrollOffset = string | number | Record<string, unknown> | null | undefined;

export class QdrantService {
  private readonly client: QdrantClient;
  private readonly sourceCollection: string;
  private readonly goldenCollection: string;
  private readonly snapshotDir: string;
  private readonly stateFile: string;
  private readonly embedder: EmbeddingService;

  constructor() {
    const host = process.env.QDRANT_HOST ?? 'localhost';
    const port = Number(process.env.QDRANT_PORT ?? 6333);
    this.client = new QdrantClient({ host, port });
    this.sourceCollection = process.env.QDRANT_COLLECTION ?? 'mrc_bot_memory';
    this.goldenCollection = process.env.GOLDEN_COLLECTION ?? 'agent_golden_knowledge';
    this.snapshotDir = process.env.SNAPSHOT_DIR ?? './snapshots';
    this.stateFile = path.join(this.snapshotDir, 'last-distill.json');
    this.embedder = new EmbeddingService();
  }

  get sourceCollectionName(): string { return this.sourceCollection; }
  get goldenCollectionName(): string { return this.goldenCollection; }

  // =================== MEMORY RETRIEVAL ===================

  async getAllMemories(agent?: string, namespace?: string): Promise<AgentMemory[]> {
    return this.scrollMemories(this.sourceCollection, agent, namespace);
  }

  async getUndistilledMemories(agent?: string, namespace?: string): Promise<AgentMemory[]> {
    // No sinceTimestamp filter — just skip already-distilled memories
    // Previously sinceTimestamp caused 5,698 old memories to be stuck forever
    return this.scrollMemories(this.sourceCollection, agent, namespace, {
      skipDistilled: true,
    });
  }

  // =================== DISTILL: ENRICH + RE-EMBED ===================

  /**
   * Mark gold as distilled + RE-EMBED with enrichedText
   * 
   * Flow per gold memory:
   * 1. Take enrichedText (LLM-written, self-contained)
   * 2. Embed(enrichedText) → new 1024-dim vector
   * 3. Update point: payload + NEW vector
   * 
   * Result: gold memory is now self-contained with matching vector
   */
  async markAsDistilledAndReembed(memories: ScoredMemory[]): Promise<{ marked: number; reembedded: number }> {
    if (memories.length === 0) return { marked: 0, reembedded: 0 };
    const now = Date.now();
    let marked = 0;
    let reembedded = 0;

    for (const memory of memories) {
      try {
        const enrichedText = memory.distilledText || memory.text;
        
        // Re-embed the enriched text
        let newVector: number[] | undefined;
        try {
          newVector = await this.embedder.embed(enrichedText);
          reembedded++;
        } catch (embedErr) {
          console.warn(`  ⚠️ Re-embed failed for ${memory.id}, keeping old vector: ${embedErr}`);
        }

        // Build payload update
        const payload: Record<string, unknown> = {
          distilled: true, distilledAt: now,
          qualityScore: memory.qualityScore, category: memory.category,
          tags: memory.tags, scoringMethod: memory.scoringMethod,
          llmReasoning: memory.llmReasoning, enrichedText: enrichedText,
        };

        if (newVector) {
          // Upsert with NEW vector + updated payload (keeps original text)
          // Need to read original point first to preserve all fields
          const existing = await this.client.retrieve(this.sourceCollection, {
            ids: [memory.id], with_payload: true, with_vector: false,
          });
          
          const existingPayload = existing?.[0]?.payload ?? {};
          const mergedPayload = { ...existingPayload, ...payload };

          await this.client.upsert(this.sourceCollection, {
            wait: true,
            points: [{
              id: memory.id,
              vector: newVector,
              payload: mergedPayload,
            }],
          });
        } else {
          // Fallback: only update payload, keep old vector
          await this.client.setPayload(this.sourceCollection, {
            points: [memory.id], payload,
          });
        }

        marked++;
      } catch (err) {
        console.error(`  ⚠️ Failed to process ${memory.id}: ${err}`);
      }
    }

    return { marked, reembedded };
  }

  /** Delete noise memories from source */
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

  // =================== STATE ===================

  async loadState(): Promise<DistillState | null> {
    try {
      const data = await readFile(this.stateFile, 'utf-8');
      return JSON.parse(data) as DistillState;
    } catch { return null; }
  }

  async saveState(state: DistillState): Promise<void> {
    await mkdir(this.snapshotDir, { recursive: true });
    await writeFile(this.stateFile, JSON.stringify(state, null, 2));
  }

  // =================== GOLDEN BACKUP ===================

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

  async upsertGoldenMemories(memories: ScoredMemory[], vectors?: Map<string, number[]>): Promise<void> {
    if (memories.length === 0) return;
    const points = memories.map((m) => ({
      id: m.id,
      vector: vectors?.get(m.id) ?? m.vector ?? new Array(1024).fill(0),
      payload: {
        text: m.text, enrichedText: m.distilledText, namespace: m.namespace,
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
        const p = (point.payload ?? {}) as Record<string, unknown>;
        memories.push({
          id: String(point.id), text: typeof p.text === 'string' ? p.text : '',
          namespace: String(p.namespace ?? ''), source_agent: String(p.source_agent ?? ''),
          source_type: String(p.source_type ?? ''), userId: String(p.userId ?? ''),
          timestamp: Number(p.timestamp ?? Date.now()),
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
    if (options?.skipDistilled) must_not.push({ key: 'distilled', match: { value: true } });
    if (options?.sinceTimestamp) must.push({ key: 'timestamp', range: { gt: options.sinceTimestamp } });
    const filter: Record<string, unknown> = {};
    if (must.length > 0) filter.must = must;
    if (must_not.length > 0) filter.must_not = must_not;
    return Object.keys(filter).length > 0 ? filter : undefined;
  }
}
