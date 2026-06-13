import { Pool } from 'pg';
import { config } from 'dotenv';
import constants from 'node:constants';

config();

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});

const URL = process.env.OLLAMA_URL || 'http://localhost:11434';

// ── Ollama embedding helper ──────────────────────────────────────────────────
async function getEmbedding(text: string): Promise<number[]> {
    const res = await fetch(`${URL}/api/embeddings`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'nomic-embed-text:v1.5',
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

// ── Insert ───────────────────────────────────────────────────────────────────
async function insertDocument(content: string): Promise<string> {
    const embedding = await getEmbedding(content);
    const vectorString = toVectorString(embedding);

    const res = await pool.query(
        `INSERT INTO documents (content, embedding, model_id)
        VALUES ($1, $2::vector, $3)
        RETURNING id`,
        [content, vectorString, 'nomic-embed-text:v1.5']
    );

    return res.rows[0].id as string;
}

// ── Search ───────────────────────────────────────────────────────────────────
async function semanticSearch(
    query: string,
    limit = 5
): Promise<{ id: string; content: string; similarity: number }[]> {
    const embedding = await getEmbedding(query);
    const queryVec = toVectorString(embedding);

    const client = await pool.connect();
    try {
        // ef_search controls recall vs speed tradeoff at query time
        // higher = more accurate, slower. Default is 40.
        await client.query('SET pgvector.ef_search = 64'); // Adjust as needed

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
    } catch (error) {
        console.error('Error during semantic search:', error);
        throw error;
    } finally {
        client.release();
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
    console.log('Inserting document...');

    const docs = [
        'Postgres is a powerful open-source relational database.',
        'Redis is an in-memory key-value store used for caching.',
        'Kafka is a distributed event streaming platform.',
        'Docker makes it easy to run applications in containers.',
        'pgvector adds vector similarity search to Postgres.',
        'Embeddings are dense numerical representations of text.',
        'HNSW is a graph-based approximate nearest neighbor algorithm.',
    ];

    for (const doc of docs) {
        const id = await insertDocument(doc);
        console.log(`Inserted: "${doc.slice(0, 40)}..." → ${id}`);
    }


    console.log('\nSearching for: "vector search in databases"\n');
    const results = await semanticSearch('vector search in databases', 3);

    results.forEach((r, i) => {
        console.log(`${i + 1}. [${r.similarity.toFixed(4)}] ${r.content}`);
    });

    await pool.end();
}

main().catch(console.error);