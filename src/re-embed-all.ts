/**
 * Re-embed ALL memories with new embedding model (qwen3-embedding:0.6b)
 * Reads text from each point, generates new vector, updates in place.
 */

import { QdrantClient } from '@qdrant/js-client-rest';

const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const EMBED_MODEL = process.env.EMBED_MODEL || 'qwen3-embedding:0.6b';
const COLLECTION = process.env.QDRANT_COLLECTION || 'mrc_bot_memory';
const BATCH_SIZE = 10;

const client = new QdrantClient({ url: QDRANT_URL });

async function embed(text: string): Promise<number[]> {
  const res = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  });
  if (!res.ok) throw new Error(`Embed failed: ${res.status}`);
  const data = await res.json() as { embeddings: number[][] };
  return data.embeddings[0];
}

async function main() {
  // 1. Create snapshot backup first
  console.log('📸 Creating backup snapshot...');
  const snapshot = await client.createSnapshot(COLLECTION);
  console.log(`✅ Snapshot: ${snapshot.name}`);

  // 2. Count total points
  const info = await client.getCollection(COLLECTION);
  const total = info.points_count ?? 0;
  console.log(`\n📊 Total points to re-embed: ${total}`);
  console.log(`🧠 New model: ${EMBED_MODEL}`);

  // 3. Scroll through ALL points and re-embed
  let processed = 0;
  let errors = 0;
  let offset: string | number | undefined = undefined;

  while (true) {
    const batch = await client.scroll(COLLECTION, {
      limit: BATCH_SIZE,
      offset,
      with_payload: ['text', 'distilledText', 'enrichedText'],
      with_vector: false,
    });

    if (batch.points.length === 0) break;

    for (const point of batch.points) {
      const payload = point.payload as Record<string, unknown>;
      // Use enriched/distilled text if available, fallback to original
      const text = (payload.enrichedText as string)
        || (payload.distilledText as string)
        || (payload.text as string)
        || '';

      if (!text || text.length < 5) {
        processed++;
        continue;
      }

      try {
        const vector = await embed(text);
        await client.updateVectors(COLLECTION, {
          points: [{ id: point.id, vector }],
        });
        processed++;
      } catch (err) {
        errors++;
        processed++;
        console.error(`  ⚠️ Error on ${point.id}: ${err}`);
      }
    }

    const pct = ((processed / total) * 100).toFixed(0);
    console.log(`  🔄 Re-embedded: ${processed}/${total} (${pct}%) [errors: ${errors}]`);

    offset = batch.next_page_offset ?? undefined;
    if (!offset) break;

    // Small delay to not overwhelm Ollama
    await new Promise(r => setTimeout(r, 200));
  }

  console.log(`\n✅ DONE! Re-embedded ${processed} points with ${EMBED_MODEL}`);
  console.log(`   Errors: ${errors}`);
}

main().catch(err => {
  console.error('❌ Fatal:', err);
  process.exit(1);
});
