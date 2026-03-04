/**
 * Embedding Service
 * - Universal safe chunking with conservative default (<= 6000 est. tokens/chunk)
 * - Adaptive retry/backoff on 400 context-length errors by shrinking chunk size
 * - Supports OpenAI-style /v1/embeddings and Ollama /api/embed|/api/embeddings
 */

const DEFAULT_MODEL = process.env.EMBED_MODEL || process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
const DEFAULT_BASE_URL = process.env.EMBED_URL || process.env.EMBEDDING_API_URL || process.env.OLLAMA_URL || 'http://localhost:11434';
const DEFAULT_TIMEOUT_MS = Number(process.env.EMBEDDING_TIMEOUT_MS || 30000);
const DEFAULT_DIMENSIONS = Number(process.env.EMBEDDING_DIMENSIONS || 1536);

const MODEL_MAX_TOKENS: Record<string, number> = {
  'text-embedding-3-small': 8192,
  'text-embedding-3-large': 8192,
  'qwen3-embedding:0.6b': 8192,
};

class EmbeddingHttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly bodyPreview: string,
    message?: string
  ) {
    super(message || `Embedding API error: ${status}`);
    this.name = 'EmbeddingHttpError';
  }
}

export interface EmbedResult {
  vector: number[];
  metadata: {
    embedding_truncated: boolean;
    embedding_original_chars: number;
    embedding_embedded_chars: number;
    embedding_token_estimate: number;
    embedding_chunks_count: number;
    embedding_safe_chunk_tokens: number;
    embedding_model: string;
  };
}

export class EmbeddingService {
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly dimensions: number;

  constructor() {
    this.model = DEFAULT_MODEL;
    this.baseUrl = DEFAULT_BASE_URL.replace(/\/+$/, '');
    this.timeoutMs = DEFAULT_TIMEOUT_MS;
    this.dimensions = DEFAULT_DIMENSIONS;
  }

  private normalizeInput(input: string): string {
    return String(input || '').trim();
  }

  private resolveEndpoints(): string[] {
    if (/(\/v1\/embeddings|\/api\/embeddings|\/api\/embed)\/?$/i.test(this.baseUrl)) {
      return [this.baseUrl];
    }
    return [`${this.baseUrl}/v1/embeddings`, `${this.baseUrl}/api/embeddings`, `${this.baseUrl}/api/embed`];
  }

  private isOpenAiEndpoint(url: string): boolean {
    return /\/v1\/embeddings\/?$/i.test(url);
  }

  private isOllamaEmbeddingsEndpoint(url: string): boolean {
    return /\/api\/embeddings\/?$/i.test(url);
  }

  private modelMaxTokens(): number {
    return MODEL_MAX_TOKENS[this.model] || 8192;
  }

  private safeChunkTokens(maxTokens: number): number {
    return Math.max(256, Math.min(6000, Math.floor(maxTokens * 0.73)));
  }

  private estimateTokens(text: string): number {
    const ws = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = Math.ceil(text.length / 4);
    return Math.max(1, Math.max(ws, chars));
  }

  private chunkText(text: string, safeTokens: number): string[] {
    const maxChars = Math.max(1, safeTokens * 4);
    if (text.length <= maxChars) return [text];

    const chunks: string[] = [];
    let cursor = 0;

    while (cursor < text.length) {
      const remaining = text.length - cursor;
      if (remaining <= maxChars) {
        const tail = text.slice(cursor).trim();
        if (tail) chunks.push(tail);
        break;
      }

      let end = cursor + maxChars;
      const window = text.slice(cursor, end);
      const naturalBreak = Math.max(
        window.lastIndexOf('\n\n'),
        window.lastIndexOf('\n'),
        window.lastIndexOf('. '),
        window.lastIndexOf('; '),
        window.lastIndexOf(' ')
      );
      if (naturalBreak > Math.floor(maxChars * 0.5)) {
        end = cursor + naturalBreak;
      }

      const chunk = text.slice(cursor, end).trim();
      if (chunk) chunks.push(chunk);
      cursor = Math.max(end, cursor + 1);
    }

    return chunks.filter(Boolean);
  }

  private l2Normalize(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    if (!Number.isFinite(norm) || norm === 0) return vector;
    return vector.map((v) => v / norm);
  }

  private weightedAverage(vectors: number[][], weights: number[]): number[] {
    if (vectors.length === 0) return [];
    const dim = vectors[0].length;
    const out = new Array<number>(dim).fill(0);
    const wSum = weights.reduce((a, b) => a + b, 0) || 1;

    for (let i = 0; i < vectors.length; i++) {
      const vec = vectors[i];
      const w = weights[i] || 1;
      for (let d = 0; d < dim; d++) out[d] += (vec[d] || 0) * w;
    }

    for (let d = 0; d < dim; d++) out[d] /= wSum;
    return this.l2Normalize(out);
  }

  private embedFromHash(text: string): number[] {
    const hash = text.split('').reduce((a, b) => {
      a = (a << 5) - a + b.charCodeAt(0);
      return a & a;
    }, 0);

    const embedding: number[] = [];
    for (let i = 0; i < this.dimensions; i++) {
      embedding.push(Math.sin(hash + i) * 0.1);
    }
    return this.l2Normalize(embedding);
  }

  private buildRequestBody(url: string, chunks: string[]): Record<string, any> {
    if (this.isOpenAiEndpoint(url)) {
      return { model: this.model, input: chunks };
    }
    if (this.isOllamaEmbeddingsEndpoint(url)) {
      return { model: this.model, prompt: chunks[0] };
    }
    return { model: this.model, input: chunks };
  }

  private extractVectors(url: string, data: any): number[][] {
    if (this.isOpenAiEndpoint(url)) {
      if (Array.isArray(data?.data)) {
        return data.data.map((d: any) => d?.embedding).filter((v: any) => Array.isArray(v));
      }
      if (Array.isArray(data?.embeddings)) {
        return data.embeddings.filter((v: any) => Array.isArray(v));
      }
      if (Array.isArray(data?.embedding)) {
        return [data.embedding];
      }
      return [];
    }

    if (this.isOllamaEmbeddingsEndpoint(url)) {
      if (Array.isArray(data?.embedding)) return [data.embedding];
      if (Array.isArray(data?.embeddings)) return data.embeddings.filter((v: any) => Array.isArray(v));
      return [];
    }

    if (Array.isArray(data?.embeddings)) return data.embeddings.filter((v: any) => Array.isArray(v));
    if (Array.isArray(data?.embedding)) return [data.embedding];
    return [];
  }

  private async postWithRetry(url: string, chunks: string[]): Promise<number[][]> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);
    const max429Retries = 4;

    try {
      for (let attempt = 0; attempt <= max429Retries; attempt++) {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.buildRequestBody(url, chunks)),
          signal: controller.signal,
        });

        if (response.status === 429 && attempt < max429Retries) {
          const backoffMs = Math.min(8000, 500 * Math.pow(2, attempt));
          await new Promise((r) => setTimeout(r, backoffMs));
          continue;
        }

        if (!response.ok) {
          const body = await response.text().catch(() => 'Unknown error');
          throw new EmbeddingHttpError(response.status, body.slice(0, 500));
        }

        const data = await response.json();
        const vectors = this.extractVectors(url, data);
        if (vectors.length === 0) throw new Error('Invalid embedding response format');
        return vectors;
      }

      throw new Error('Embedding API 429 retries exhausted');
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        throw new Error('Embedding request timed out');
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async embedChunks(chunks: string[]): Promise<number[][]> {
    if (chunks.length === 0) throw new Error('No chunks to embed');

    const endpoints = this.resolveEndpoints();
    let lastError: Error | null = null;

    for (const endpoint of endpoints) {
      const openAi = this.isOpenAiEndpoint(endpoint);
      const ollamaEmbeddings = this.isOllamaEmbeddingsEndpoint(endpoint);

      try {
        if (!openAi && !ollamaEmbeddings && chunks.length > 1) {
          const vectors: number[][] = [];
          for (const chunk of chunks) {
            const single = await this.postWithRetry(endpoint, [chunk]);
            vectors.push(single[0]);
          }
          return vectors;
        }

        const vectors = await this.postWithRetry(endpoint, chunks);
        if (vectors.length !== chunks.length) {
          throw new Error(`Embedding vector count mismatch: expected=${chunks.length}, got=${vectors.length}`);
        }
        return vectors;
      } catch (error: any) {
        lastError = error;

        const isContext400 =
          error instanceof EmbeddingHttpError &&
          error.status === 400 &&
          /context length|maximum context|too many tokens|exceed|8192|token/i.test(
            error.bodyPreview || error.message || ''
          );

        if (isContext400) {
          // Bubble up so adaptive shrink loop can immediately retry smaller chunks.
          throw error;
        }

        if (
          error instanceof EmbeddingHttpError &&
          (error.status === 404 || error.status === 405 || error.status === 429)
        ) {
          continue;
        }
        if (endpoint !== endpoints[endpoints.length - 1]) continue;
      }
    }

    throw lastError || new Error('Embedding API error: no endpoint succeeded');
  }

  /** Embed a single text with adaptive chunk-shrink retries on context-length 400 */
  async embed(text: string): Promise<EmbedResult> {
    const normalized = this.normalizeInput(text);
    const originalChars = normalized.length;

    if (!normalized) {
      return {
        vector: this.embedFromHash(''),
        metadata: {
          embedding_truncated: false,
          embedding_original_chars: 0,
          embedding_embedded_chars: 0,
          embedding_token_estimate: 0,
          embedding_chunks_count: 0,
          embedding_safe_chunk_tokens: this.safeChunkTokens(this.modelMaxTokens()),
          embedding_model: this.model,
        },
      };
    }

    const maxTokens = this.modelMaxTokens();
    const baseSafe = this.safeChunkTokens(maxTokens);
    const shrinkFactors = [1, 0.8, 0.65, 0.5, 0.4, 0.3];

    for (const factor of shrinkFactors) {
      const safeTokens = Math.max(256, Math.floor(baseSafe * factor));
      const chunks = this.chunkText(normalized, safeTokens);
      const weights = chunks.map((c) => this.estimateTokens(c));

      try {
        const vectors = await this.embedChunks(chunks);
        const vector = vectors.length === 1 ? this.l2Normalize(vectors[0]) : this.weightedAverage(vectors, weights);
        return {
          vector,
          metadata: {
            embedding_truncated: chunks.length > 1,
            embedding_original_chars: originalChars,
            embedding_embedded_chars: chunks.reduce((sum, c) => sum + c.length, 0),
            embedding_token_estimate: chunks.reduce((sum, c) => sum + this.estimateTokens(c), 0),
            embedding_chunks_count: chunks.length,
            embedding_safe_chunk_tokens: safeTokens,
            embedding_model: this.model,
          },
        };
      } catch (error: any) {
        const isContext400 =
          error instanceof EmbeddingHttpError &&
          error.status === 400 &&
          /context length|maximum context|too many tokens|exceed|8192|token/i.test(error.bodyPreview || error.message || '');

        if (isContext400) {
          console.warn(`  ⚠️ Context-length 400, retrying smaller chunks (${safeTokens} tokens)`);
          continue;
        }

        console.warn(`  ⚠️ Embedding API failed, fallback hash vector: ${error.message}`);
        return {
          vector: this.embedFromHash(normalized),
          metadata: {
            embedding_truncated: true,
            embedding_original_chars: originalChars,
            embedding_embedded_chars: normalized.length,
            embedding_token_estimate: this.estimateTokens(normalized),
            embedding_chunks_count: chunks.length,
            embedding_safe_chunk_tokens: safeTokens,
            embedding_model: this.model,
          },
        };
      }
    }

    console.warn('  ⚠️ Exhausted context retries, fallback hash vector');
    return {
      vector: this.embedFromHash(normalized),
      metadata: {
        embedding_truncated: true,
        embedding_original_chars: originalChars,
        embedding_embedded_chars: normalized.length,
        embedding_token_estimate: this.estimateTokens(normalized),
        embedding_chunks_count: this.chunkText(normalized, baseSafe).length,
        embedding_safe_chunk_tokens: baseSafe,
        embedding_model: this.model,
      },
    };
  }

  async embedBatch(texts: string[]): Promise<EmbedResult[]> {
    const results: EmbedResult[] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }
}
