**Result 1 — Semantic search is correct**

```
1. [0.6687] pgvector adds vector similarity search to Postgres.
2. [0.5910] Postgres is a powerful open-source relational database.
3. [0.5138] Redis is an in-memory key-value store used for caching.
```

Query was `"vector search in databases"`. The model correctly ranked pgvector #1 (exact topic), Postgres #2 (database, same ecosystem), Redis #3 (also a database). HNSW and Embeddings didn't even rank — they're about algorithms/ML, not "databases" specifically. The embedding model understood semantic intent, not keyword overlap. This is working exactly right.

---

**Result 2 — Seq Scan is expected, not a bug**

The planner output says:

```
->  Seq Scan on documents  (cost=0.00..17.88 rows=450 width=32)
```

Two reasons this happened:

**Reason 1 — Only 7 rows.** The HNSW index has overhead: it's a graph structure, traversing nodes takes work. For 7 rows that fit in a single memory page, a sequential scan literally is faster. The planner is correct. It would be stupid to use the index here.

**Reason 2 — Dimension mismatch in your test vector.** You passed `[0.1,0.2,0.3]` which is a 3-dim vector. Your column is `vector(768)`. pgvector can't use the index because dimensions don't match. EXPLAIN doesn't execute the query so it didn't error — but actual execution with that vector would fail with a dimension error.

---

**To actually see the index being used — two steps**

First, use a real 768-dim vector from your table. Inside psql:

```sql
-- Pull an actual embedding from your table to use as the query vector
SELECT embedding FROM documents LIMIT 1;
```

That'll print a massive array. Instead of copy-pasting it, do this:

```sql
-- Use a subquery so pgvector sees correct dims
EXPLAIN
SELECT id, 1 - (embedding <=> (SELECT embedding FROM documents LIMIT 1)) AS similarity
FROM documents
ORDER BY embedding <=> (SELECT embedding FROM documents LIMIT 1)
LIMIT 5;
```

Still shows Seq Scan? Because 7 rows. Force the planner:

```sql
SET enable_seqscan = off;

EXPLAIN
SELECT id, 1 - (embedding <=> (SELECT embedding FROM documents LIMIT 1)) AS similarity
FROM documents
ORDER BY embedding <=> (SELECT embedding FROM documents LIMIT 1)
LIMIT 5;
```

Now you'll see:

```
Index Scan using idx_docs_embedding_hnsw on documents
```

`enable_seqscan = off` is just a planner hint for testing — never use it in production. In real usage the index kicks in automatically once you have enough rows.

---

**Bulk insert to see real planner behavior**

Add this to your `src/index.ts` and run it:

```typescript
async function bulkInsert() {
  const fakeDocs = Array.from({ length: 200 }, (_, i) => 
    `Document ${i}: ${['machine learning', 'databases', 'networking', 'security', 'cloud'][i % 5]} concepts explained.`
  );

  console.log('Bulk inserting 200 documents...');
  for (const doc of fakeDocs) {
    await insertDocument(doc);
  }
  console.log('Done.');
}

// call bulkInsert() instead of main() temporarily
bulkInsert().then(() => pool.end());
```

After this runs, back in psql:

```sql
-- Reset seqscan (back to default)
SET enable_seqscan = on;

EXPLAIN
SELECT id, 1 - (embedding <=> (SELECT embedding FROM documents LIMIT 1)) AS similarity
FROM documents
ORDER BY embedding <=> (SELECT embedding FROM documents LIMIT 1)
LIMIT 5;
```

With 207 rows, the planner will likely switch to the HNSW index on its own. The exact threshold depends on Postgres cost estimates — generally somewhere in the hundreds of rows range for vector indexes.

---

**Mental model to lock in**

The index isn't for correctness, it's for scale. With 7 rows, brute force beats HNSW. With 1M rows, HNSW beats brute force by orders of magnitude. The same query, same index, same code — the planner decides which path is cheaper based on row count and cost estimates. That's the whole point of ANN: you trade a tiny bit of recall for massive speed gains at scale.

Run the bulk insert and share what EXPLAIN shows after.