/**
 * Embedding Service — calls Ollama qwen3-embedding:0.6b locally
 * Same model OpenClaw uses to embed memories
 * 
 * Re-embeds enrichedText so vector matches the richer content
 */

const DEFAULT_MODEL = 'qwen3-embedding:0.6b';
const DEFAULT_OLLAMA_URL = 'http://localhost:11434';

export class EmbeddingService {
  private readonly model: string;
  private readonly ollamaUrl: string;

  constructor() {
    this.model = process.env.EMBED_MODEL ?? DEFAULT_MODEL;
    this.ollamaUrl = process.env.OLLAMA_URL ?? DEFAULT_OLLAMA_URL;
  }

  /** Embed a single text → 1024-dim vector */
  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.ollamaUrl}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: text }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embed error: ${response.status} ${await response.text()}`);
    }

    const data = await response.json() as { embeddings: number[][] };
    if (!data.embeddings?.[0]?.length) {
      throw new Error('Empty embedding response');
    }

    return data.embeddings[0];
  }

  /** Embed batch of texts — sequential to avoid overloading Ollama */
  async embedBatch(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }
}
