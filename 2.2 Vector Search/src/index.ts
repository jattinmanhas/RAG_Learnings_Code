import { Pool } from 'pg';
import { config } from 'dotenv';

config();

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});

const URL = process.env.OLLAMA_URL || 'http://localhost:11434';

const EMBEDDING_MODEL = 'nomic-embed-text:v1.5';

// ── Ollama embedding helper ──────────────────────────────────────────────────
async function getEmbedding(text: string): Promise<number[]> {
    const res = await fetch(`${URL}/api/embeddings`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: EMBEDDING_MODEL,
            prompt: text,
        }),
    });

    if (!res.ok) {
        const body = await res.text();
        throw new Error(`Ollama API error ${res.status}: ${body}`);
    }

    const data = await res.json() as { embedding: number[] };
    return data.embedding;
}

// pgvector expects the vector as a string: '[0.1,0.2,...]'
function toVectorString(embedding: number[]): string {
    return `[${embedding.join(',')}]`;
}

// ── Schema ────────────────────────────────────────────────────────────────────
// Idempotent setup so the demo is self-contained. Mirrors src/pgsql.sql, plus a
// GIN index on metadata to make the filtered-search example fast.
async function ensureSchema(): Promise<void> {
    await pool.query(`CREATE EXTENSION IF NOT EXISTS vector`);
    await pool.query(`
        CREATE TABLE IF NOT EXISTS documents (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content    TEXT NOT NULL,
            metadata   JSONB DEFAULT '{}',
            embedding  vector(768),
            model_id   TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    `);
    // vector_cosine_ops → the <=> (cosine distance) operator uses this index.
    // Each distance metric needs its own opclass/index.
    await pool.query(`
        CREATE INDEX IF NOT EXISTS idx_docs_embedding_hnsw
        ON documents USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    `);
    await pool.query(`
        CREATE INDEX IF NOT EXISTS idx_docs_metadata
        ON documents USING gin (metadata)
    `);
}

// ── Insert ───────────────────────────────────────────────────────────────────
async function insertDocument(
    content: string,
    metadata: Record<string, unknown> = {}
): Promise<string> {
    const embedding = await getEmbedding(content);
    const vectorString = toVectorString(embedding);

    const res = await pool.query(
        `INSERT INTO documents (content, embedding, metadata, model_id)
        VALUES ($1, $2::vector, $3, $4)
        RETURNING id`,
        [content, vectorString, metadata, EMBEDDING_MODEL]
    );

    return res.rows[0].id as string;
}

type SearchResult = { id: string; content: string; similarity: number };

// ── 1. Semantic search ────────────────────────────────────────────────────────
async function semanticSearch(query: string, limit = 5): Promise<SearchResult[]> {
    const embedding = await getEmbedding(query);
    const queryVec = toVectorString(embedding);

    const client = await pool.connect();
    try {
        // hnsw.ef_search is the query-time recall/speed dial: it bounds how many
        // candidate nodes the HNSW graph walk keeps. Higher = more accurate,
        // slower. Default is 40. It's a per-session GUC, so set it on the same
        // connection the query runs on.
        //
        // NOTE: the parameter is `hnsw.ef_search`, NOT `pgvector.ef_search`.
        // Postgres silently accepts unknown dotted names as custom variables,
        // so a typo here is a no-op with no error — ef_search stays at 40.
        await client.query('SET hnsw.ef_search = 64');

        const result = await client.query(
            `SELECT
                id,
                content,
                1 - (embedding <=> $1::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> $1::vector
            LIMIT $2`,
            [queryVec, limit]
        );

        return result.rows;
    } finally {
        client.release();
    }
}

// ── 2. Distance metrics side by side ──────────────────────────────────────────
// pgvector has three operators, all "lower = closer":
//   <=>  cosine distance          similarity = 1 - (a <=> b)
//   <->  Euclidean / L2 distance  straight-line distance
//   <#>  negative inner product   returns -(a · b), so more-negative = closer
// Match the operator to how your model was trained. nomic-embed-text returns
// normalized vectors, so cosine and inner product rank identically.
async function compareDistanceMetrics(query: string): Promise<void> {
    const queryVec = toVectorString(await getEmbedding(query));

    const metrics: { name: string; expr: string }[] = [
        { name: 'cosine    (<=>)', expr: 'embedding <=> $1::vector' },
        { name: 'L2        (<->)', expr: 'embedding <-> $1::vector' },
        { name: 'inner prod(<#>)', expr: 'embedding <#> $1::vector' },
    ];

    console.log(`Top match per distance metric for "${query}":`);
    for (const m of metrics) {
        const res = await pool.query(
            `SELECT content, ${m.expr} AS distance
             FROM documents ORDER BY ${m.expr} LIMIT 1`,
            [queryVec]
        );
        const row = res.rows[0];
        console.log(`  ${m.name}  dist=${Number(row.distance).toFixed(4)}  ${row.content}`);
    }
}

// ── 3. Filtered (hybrid) search ───────────────────────────────────────────────
// Real RAG is rarely "search everything" — it's "search this tenant / language /
// document set". Combine a structured metadata predicate with vector ranking in
// one query. `metadata @> $2` (JSONB containment) is backed by the GIN index.
async function filteredSearch(
    query: string,
    filter: Record<string, unknown>,
    limit = 5
): Promise<SearchResult[]> {
    const queryVec = toVectorString(await getEmbedding(query));

    const res = await pool.query(
        `SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
         FROM documents
         WHERE metadata @> $2
         ORDER BY embedding <=> $1::vector
         LIMIT $3`,
        [queryVec, filter, limit]
    );
    return res.rows;
}

// ── 4. ef_search recall vs speed, live ────────────────────────────────────────
// Same nearest-neighbor query at several ef_search values. At small scale the
// timings are noisy, but the shape is the lesson: raising ef_search widens the
// graph search → higher recall, more work.
async function demoEfSearch(query: string): Promise<void> {
    const queryVec = toVectorString(await getEmbedding(query));

    for (const ef of [10, 40, 100]) {
        const client = await pool.connect();
        try {
            await client.query(`SET hnsw.ef_search = ${ef}`);
            const start = performance.now();
            const res = await client.query(
                `SELECT content FROM documents ORDER BY embedding <=> $1::vector LIMIT 1`,
                [queryVec]
            );
            const ms = (performance.now() - start).toFixed(2);
            console.log(`  ef_search=${String(ef).padEnd(3)}  ${ms.padStart(6)}ms  top: ${res.rows[0].content}`);
        } finally {
            client.release();
        }
    }
}

// ── 5. Query plan (index vs seq scan) ─────────────────────────────────────────
// With only a handful of rows you'll see a Seq Scan — brute force genuinely
// beats walking an HNSW graph at tiny scale. That's correct, not a bug. The
// index earns its keep once the table grows into the hundreds/thousands of rows.
async function explainSearch(query: string): Promise<void> {
    const queryVec = toVectorString(await getEmbedding(query));
    const res = await pool.query(
        `EXPLAIN
         SELECT id, content FROM documents
         ORDER BY embedding <=> $1::vector
         LIMIT 5`,
        [queryVec]
    );
    console.log('EXPLAIN plan:');
    for (const row of res.rows) {
        console.log(`  ${row['QUERY PLAN']}`);
    }
}

// ── Seed ──────────────────────────────────────────────────────────────────────
// Insert a small corpus only when the table is empty, so re-runs are cheap.
// Each doc carries a category so the filtered-search example has something to
// constrain on.
async function seed(): Promise<void> {
    const { rows } = await pool.query('SELECT COUNT(*)::int AS count FROM documents');
    if (rows[0].count > 0) {
        console.log(`Corpus already has ${rows[0].count} documents, skipping seed.\n`);
        return;
    }

    const docs: { content: string; category: string }[] = [
        { content: 'Postgres is a powerful open-source relational database.', category: 'database' },
        { content: 'Redis is an in-memory key-value store used for caching.', category: 'database' },
        { content: 'Kafka is a distributed event streaming platform.', category: 'infra' },
        { content: 'Docker makes it easy to run applications in containers.', category: 'infra' },
        { content: 'pgvector adds vector similarity search to Postgres.', category: 'database' },
        { content: 'Embeddings are dense numerical representations of text.', category: 'ml' },
        { content: 'HNSW is a graph-based approximate nearest neighbor algorithm.', category: 'ml' },
    ];

    console.log('Seeding corpus...');
    for (const d of docs) {
        const id = await insertDocument(d.content, { category: d.category });
        console.log(`  inserted ${id.slice(0, 8)}  (${d.content.slice(0, 40)})`);
    }
    console.log();
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
    await ensureSchema();
    await seed();

    console.log('── 1. Semantic search ─────────────────────────────');
    const results = await semanticSearch('vector search in databases', 3);
    results.forEach((r, i) => {
        console.log(`  ${i + 1}. [${r.similarity.toFixed(4)}] ${r.content}`);
    });

    console.log('\n── 2. Distance metrics ────────────────────────────');
    await compareDistanceMetrics('vector search in databases');

    console.log('\n── 3. Filtered (hybrid) search ────────────────────');
    const filtered = await filteredSearch('fast lookups', { category: 'database' }, 3);
    filtered.forEach((r, i) => {
        console.log(`  ${i + 1}. [${r.similarity.toFixed(4)}] ${r.content}`);
    });

    console.log('\n── 4. ef_search recall vs speed ───────────────────');
    await demoEfSearch('distributed systems and streaming');

    console.log('\n── 5. Query plan (index vs seq scan) ──────────────');
    await explainSearch('vector search in databases');

    await pool.end();
}

main().catch(async (err) => {
    console.error(err);
    await pool.end();
    process.exit(1);
});
