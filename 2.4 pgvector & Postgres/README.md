# 2.4 pgvector & Postgres

Using Postgres as the vector database for a RAG system, via the **pgvector**
extension. This folder is a commented, runnable reference — not an app.

- `docker-compose.yml` — Postgres 16 with pgvector preinstalled.
- `pgvector-reference.sql` — every query and concept you need, heavily commented.

## Quick start

```bash
# 1. Start the database
docker compose up -d

# 2. Open a psql shell
docker compose exec postgres psql -U user -d ragdb

# 3. Inside psql, paste sections of pgvector-reference.sql as you learn them,
#    or run the whole file:
docker compose exec -T postgres psql -U user -d ragdb < pgvector-reference.sql

# Stop (keep data):   docker compose down
# Stop (wipe data):   docker compose down -v
```

## The mental model

```
question ──embed──▶ query vector ──▶  SELECT ... ORDER BY embedding <=> :q LIMIT k
                                       (find nearest chunks)  ──▶  feed to LLM
```

The whole game is: store one **vector per chunk** (next to its text), then at
query time find the nearest vectors to the question's embedding.

## What the reference covers

1. Enabling the extension
2. Designing the table (one row = one chunk)
3. The three distance operators (`<=>` cosine, `<#>` inner product, `<->` L2)
4. Inserting vectors (single + batch)
5. The core top-k RAG search query
6. Metadata filtering + similarity thresholds
7. ANN indexes — HNSW vs IVFFlat, and when to use each
8. Query-time recall/speed tuning (`hnsw.ef_search`, `ivfflat.probes`)
9. Verifying the index is used with `EXPLAIN ANALYZE`
10. Maintenance & inspection queries
11. Production gotchas (dimensions are forever, never mix models, etc.)

## Most important takeaways

- **Cosine distance (`<=>`) is the default** for text-embedding RAG.
- **`similarity = 1 - (embedding <=> query)`** — but `ORDER BY` the raw
  distance so the index can be used.
- **The index op-class must match the operator** (`vector_cosine_ops` ↔ `<=>`).
- **Dimensions must match your model exactly** and changing models means
  re-embedding everything — store a `model_id` per row.
