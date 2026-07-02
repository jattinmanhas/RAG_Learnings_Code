import OpenAI from "openai";

const client = new OpenAI({
    baseURL: "http://localhost:11434/v1",
    apiKey: "ollama" // Required by the library but ignored by Ollama
});

const EMBEDDING_MODEL = 'nomic-embed-text:v1.5'; // or 'text-embedding-3-small'


async function main() {
    const response = await client.embeddings.create({
        model: EMBEDDING_MODEL,
        input: "The quick brown fox jumps over the lazy dog.",
        encoding_format: "float"
    });

    const first = response.data[0];
    if (!first) throw new Error("No embedding returned");

    console.log(first.embedding.slice(0, 5)); // first 5 dims
    console.log("Dimensions:", first.embedding.length);
}

// Most providers cap inputs per request (OpenAI ~2048, self-hosted far less).
// Chunk large workloads to stay under the limit and bound request size.
const MAX_BATCH_SIZE = 256;

// Retry transient failures (429 rate limits, 5xx, network blips) with
// exponential backoff. Never retry other 4xx — those are request bugs that
// fail identically every time.
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 5): Promise<T> {
    let lastErr: unknown;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (err: any) {
            const status = err?.status ?? err?.response?.status;
            const retryable = status === 429 || status === undefined || status >= 500;
            if (!retryable) throw err;
            lastErr = err;
            const backoff = 2 ** attempt * 200; // 200ms, 400, 800, ...
            await new Promise(r => setTimeout(r, backoff));
        }
    }
    throw lastErr;
}

// Embeddings are deterministic for a given (model, text) — ideal to cache.
// This saves real money on re-ingested docs and repeated queries. Swap the Map
// for Redis to share the cache across instances.
const embeddingCache = new Map<string, number[]>();
const cacheKey = (text: string) => `${EMBEDDING_MODEL}\0${text}`;

// Batch embedding — always prefer this over looping single calls.
// Handles caching + chunking so callers just pass any-sized array.
async function batchEmbedding(texts: string[]): Promise<number[][]> {
    const results: number[][] = new Array(texts.length);
    const missIdx: number[] = [];
    const missText: string[] = [];

    texts.forEach((t, i) => {
        const cached = embeddingCache.get(cacheKey(t));
        if (cached) results[i] = cached;
        else { missIdx.push(i); missText.push(t); }
    });

    for (let start = 0; start < missText.length; start += MAX_BATCH_SIZE) {
        const slice = missText.slice(start, start + MAX_BATCH_SIZE);
        const response = await withRetry(() => client.embeddings.create({
            model: EMBEDDING_MODEL,
            input: slice,
            encoding_format: 'float',
        }));

        response.data
            .sort((a, b) => a.index - b.index) // spec doesn't guarantee order
            .forEach((item, j) => {
                const idx = missIdx[start + j]!;
                results[idx] = item.embedding;
                embeddingCache.set(cacheKey(slice[j]!), item.embedding);
            });
    }
    return results;
}

main();

// When vectors are normalized, dot product = cosine similarity
const similarity = (a : number[], b: number[]) => {
    const dot = a.reduce((sum, val, i) => sum + val * (b[i] ?? 0), 0);
    return dot; // if normalized, this is cosine similarity
}

// Full cosine — stays correct for ANY model, even ones that don't normalize.
// Use this as the safe default; fall back to `similarity` (plain dot) only
// when you KNOW the vectors are unit-length.
const cosineSimilarity = (a: number[], b: number[]) => {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i]! * (b[i] ?? 0);
        normA += a[i]! * a[i]!;
        normB += (b[i] ?? 0) * (b[i] ?? 0);
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const normalize = (v: number[]) => {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// Matryoshka truncation — nomic-embed-text & text-embedding-3 pack most of the
// signal into the first N dims. Truncating (then re-normalizing) shrinks
// storage and speeds up search at a small accuracy cost. A real lever once you
// have millions of vectors.
const truncateDims = (v: number[], dims: number) =>
    dims >= v.length ? v : normalize(v.slice(0, dims));

const sentences = [
    'A dog ran across the yard',
    'A puppy sprinted through the garden',   // should be similar to [0]
    'The stock market fell sharply today',    // should be dissimilar
    'Equity markets experienced a steep decline', // should be similar to [2]
];

async function testBatchEmbedding() {
    const embeddings = await batchEmbedding(sentences);

      // Compare all pairs
    for (let i = 0; i < sentences.length; i++) {
        for (let j = i + 1; j < sentences.length; j++) {
            const a = embeddings[i];
            const b = embeddings[j];
            if (!a || !b) continue;
            const sim = cosineSimilarity(a, b);
            console.log(`Similarity between "${sentences[i]}" and "${sentences[j]}": ${sim.toFixed(4)}`);
        }
    }

    // Matryoshka: compare full-dim vs 256-dim similarity for the first pair.
    const [a, b] = [embeddings[0], embeddings[1]];
    if (a && b) {
        const full = cosineSimilarity(a, b);
        const small = cosineSimilarity(truncateDims(a, 256), truncateDims(b, 256));
        console.log(`\nPair [0]x[1] similarity: full=${full.toFixed(4)} 256-dim=${small.toFixed(4)}`);
    }

    // Second pass is served entirely from the embedding cache (near-instant).
    const t0 = Date.now();
    await batchEmbedding(sentences);
    console.log(`Cached re-embed took ${Date.now() - t0}ms`);
}

testBatchEmbedding();

//The .sort() by index — good instinct adding that. The OpenAI spec doesn't guarantee response order matches input order, so that's a production-correctness thing, not just cleanliness.
//The dot product = cosine similarity comment — this only holds because nomic-embed-text:v1.5 outputs L2-normalized vectors (unit vectors). If you switched to a model that doesn't normalize, you'd get wrong results silently.
//The batch embedding function is a great example of how to efficiently get embeddings for multiple inputs. Always prefer batching over looping single calls for performance and cost reasons.