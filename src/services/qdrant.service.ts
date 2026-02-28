import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { QdrantClient } from '@qdrant/js-client-rest';
import { AgentMemory, ScoredMemory } from '../types';

const DEFAULT_BATCH_SIZE = 256;

type ScrollOffset = string | number | Record<string, unknown> | null | undefined;

export class QdrantService {
  private readonly client: QdrantClient;
  private readonly sourceCollection: string;
  private readonly goldenCollection: string;
  private readonly snapshotDir: string;

  constructor() {
    const host = process.env.QDRANT_HOST ?? 'localhost';
    const port = Number(process.env.QDRANT_PORT ?? 6333);

    this.client = new QdrantClient({ host, port });
    this.sourceCollection = process.env.QDRANT_COLLECTION ?? 'mrc_bot_memory';
    this.goldenCollection = process.env.GOLDEN_COLLECTION ?? 'agent_golden_knowledge';
    this.snapshotDir = process.env.SNAPSHOT_DIR ?? './snapshots';
  }

  get sourceCollectionName(): string {
    return this.sourceCollection;
  }

  get goldenCollectionName(): string {
    return this.goldenCollection;
  }

  async getAllMemories(agent?: string, namespace?: string): Promise<AgentMemory[]> {
    const memories: AgentMemory[] = [];
    let nextPageOffset: ScrollOffset = null;

    const filter = this.buildFilter(agent, namespace);

    do {
      const res = await this.client.scroll(this.sourceCollection, {
        limit: DEFAULT_BATCH_SIZE,
        offset: (nextPageOffset ?? undefined) as never,
        with_payload: true,
        with_vector: true,
        filter,
      });

      for (const point of res.points) {
        const payload = (point.payload ?? {}) as Record<string, unknown>;
        const text = typeof payload.text === 'string' ? payload.text : '';

        memories.push({
          id: String(point.id),
          text,
          namespace: String(payload.namespace ?? ''),
          source_agent: String(payload.source_agent ?? ''),
          source_type: String(payload.source_type ?? ''),
          userId: String(payload.userId ?? ''),
          timestamp: Number(payload.timestamp ?? Date.now()),
          vector: Array.isArray(point.vector) ? (point.vector as number[]) : undefined,
        });
      }

      nextPageOffset = res.next_page_offset;
    } while (nextPageOffset !== null && nextPageOffset !== undefined);

    return memories;
  }

  async createGoldenCollection(): Promise<void> {
    const exists = await this.client.collectionExists(this.goldenCollection);
    if (exists.exists) return;

    const sourceInfo = await this.client.getCollection(this.sourceCollection);
    const sourceVectors = sourceInfo.config?.params?.vectors;

    if (!sourceVectors) {
      await this.client.createCollection(this.goldenCollection, {
        vectors: { size: 1024, distance: 'Cosine' },
      });
      return;
    }

    await this.client.createCollection(this.goldenCollection, {
      vectors: sourceVectors,
    });
  }

  async upsertGoldenMemories(memories: ScoredMemory[]): Promise<void> {
    if (memories.length === 0) return;

    const points = memories.map((memory) => ({
      id: memory.id,
      vector: memory.vector ?? new Array(1024).fill(0),
      payload: {
        text: memory.text,
        distilledText: memory.distilledText,
        namespace: memory.namespace,
        source_agent: memory.source_agent,
        source_type: memory.source_type,
        userId: memory.userId,
        timestamp: memory.timestamp,
        qualityScore: memory.qualityScore,
        category: memory.category,
        tags: memory.tags,
        scoringMethod: memory.scoringMethod,
        llmReasoning: memory.llmReasoning,
      },
    }));

    const chunkSize = 128;
    for (let i = 0; i < points.length; i += chunkSize) {
      const chunk = points.slice(i, i + chunkSize);
      await this.client.upsert(this.goldenCollection, {
        wait: true,
        points: chunk,
      });
    }
  }

  async createSnapshot(collectionName: string): Promise<string> {
    const result = await this.client.createSnapshot(collectionName);

    if (!result?.name) {
      throw new Error(`Failed to create snapshot for collection: ${collectionName}`);
    }

    await mkdir(this.snapshotDir, { recursive: true });
    return path.join(this.snapshotDir, result.name);
  }

  async listSnapshots(collectionName: string): Promise<string[]> {
    const snapshots = await this.client.listSnapshots(collectionName);
    return snapshots.map((snapshot) => snapshot.name);
  }

  async getCollectionInfo(name: string) {
    return this.client.getCollection(name);
  }

  private buildFilter(agent?: string, namespace?: string) {
    const must: Array<Record<string, unknown>> = [];

    if (agent) {
      must.push({ key: 'source_agent', match: { value: agent } });
    }

    if (namespace) {
      must.push({ key: 'namespace', match: { value: namespace } });
    }

    return must.length > 0 ? { must } : undefined;
  }
}
