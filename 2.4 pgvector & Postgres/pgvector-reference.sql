-- ============================================================================
--  pgvector FOR RAG — A COMMENTED REFERENCE OF EVERY QUERY YOU ACTUALLY NEED
-- ============================================================================
--
--  WHAT IS pgvector?
--  -----------------
--  pgvector is a Postgres EXTENSION that adds a `vector` column type plus
--  distance operators and ANN (Approximate Nearest Neighbor) indexes. It lets
--  your existing Postgres database double as a vector database — so your
--  embeddings live right next to your normal relational data (no separate
--  service like Pinecone/Weaviate to run).
--
--  WHERE IT FITS IN A RAG PIPELINE
--  -------------------------------
--    extract text -> chunk -> EMBED each chunk into a vector
--      -> STORE vectors here (INSERT) -> at query time, embed the user's
--      question and find the nearest chunks (SELECT ... ORDER BY <distance>)
--      -> feed those chunks to the LLM as context.
--
--  HOW TO RUN THIS FILE
--    docker compose up -d
--    docker compose exec postgres psql -U user -d ragdb -f /path/inside/container.sql
--  ...or just open psql and paste sections as you learn them.
-- ============================================================================


-- ============================================================================
--  1. ENABLE THE EXTENSION  (run once per database)
-- ============================================================================
-- Without this, the `vector` type does not exist. IF NOT EXISTS makes it safe
-- to re-run. The pgvector Docker image has the files installed, but every
-- database still needs this CREATE EXTENSION to activate them.
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it's active and check the version (newer = more features/index types):
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';


-- ============================================================================
--  2. CREATE THE TABLE
-- ============================================================================
-- The golden rule: ONE ROW = ONE CHUNK. You do NOT store a whole document in
-- one row — you store each chunk you want to retrieve independently.
CREATE TABLE IF NOT EXISTS documents (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- The chunk's text. You return THIS to the LLM as context, so always keep
  -- the original text alongside its vector. The vector is only for *finding*
  -- the chunk; the LLM reads the text.
  content    TEXT NOT NULL,

  -- Free-form JSON for filtering & provenance: source file name, page number,
  -- chunk index, author, tags, timestamps... JSONB (not JSON) is stored binary
  -- and is indexable/queryable. This is how you do "search only within these
  -- documents" later.
  metadata   JSONB DEFAULT '{}',

  -- THE EMBEDDING. The number in vector(N) is the DIMENSION and it MUST match
  -- your embedding model's output EXACTLY and forever:
  --    nomic-embed-text         -> 768
  --    text-embedding-3-small   -> 1536
  --    text-embedding-3-large   -> 3072
  -- If you change models, the dimension changes and you must re-embed
  -- everything. (Note: a plain `vector` column maxes at 2000 dims for an index;
  -- for >2000 dims you need `halfvec` — see section 11.)
  embedding  vector(768),

  -- Store WHICH model produced the embedding. Critical for migrations: you can
  -- never mix vectors from different models in the same similarity search, and
  -- this column lets you find/rebuild stale rows after a model change.
  model_id   TEXT NOT NULL,

  created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================================
--  3. THE THREE DISTANCE OPERATORS  (the heart of pgvector)
-- ============================================================================
-- pgvector compares two vectors with these operators. SMALLER distance =
-- MORE similar. Pick the one that matches how your embedding model was trained.
--
--    <=>   COSINE distance        = 1 - cosine_similarity   (most common for
--                                   RAG; ignores vector length, compares angle)
--    <#>   NEGATIVE inner product  (dot product, negated so "smaller = closer")
--    <->   EUCLIDEAN / L2 distance (straight-line distance)
--    <+>   L1 / Manhattan distance (pgvector >= 0.7)
--
-- IMPORTANT: cosine_similarity = 1 - (a <=> b). So to turn distance into a
-- human-friendly "similarity score" you subtract from 1 (see queries below).
--
-- WHICH ONE? Use what your model's docs recommend. Most modern text models
-- (OpenAI, nomic) are trained for cosine / normalized vectors, so <=> is the
-- safe default. If your vectors are L2-normalized, cosine and inner product
-- rank identically.
SELECT
  '[1,2,3]'::vector <=> '[2,3,4]'::vector AS cosine_distance,
  '[1,2,3]'::vector <#> '[2,3,4]'::vector AS neg_inner_product,
  '[1,2,3]'::vector <-> '[2,3,4]'::vector AS l2_distance;


-- ============================================================================
--  4. INSERTING VECTORS
-- ============================================================================
-- A vector literal is just text: '[0.1, 0.2, 0.3, ...]'. From application code
-- you build that string (or use a driver/ORM that does it). Examples here use
-- tiny 3-dim vectors for readability — in reality these are 768+ numbers.

-- Single insert:
INSERT INTO documents (content, metadata, embedding, model_id)
VALUES (
  'pgvector adds vector similarity search to Postgres.',
  '{"source": "intro.md", "chunk": 0}',
  '[0.10, 0.20, 0.30]',           -- pretend this is a 768-dim embedding
  'nomic-embed-text:v1.5'
);

-- BATCH insert — ALWAYS prefer this over a loop of single inserts. One round
-- trip instead of N; massively faster when loading thousands of chunks.
INSERT INTO documents (content, metadata, embedding, model_id)
VALUES
  ('Postgres is a powerful open-source relational database.',
   '{"source": "intro.md", "chunk": 1}', '[0.11, 0.19, 0.31]', 'nomic-embed-text:v1.5'),
  ('Redis is an in-memory key-value store used for caching.',
   '{"source": "intro.md", "chunk": 2}', '[0.80, 0.10, 0.05]', 'nomic-embed-text:v1.5'),
  ('HNSW is a graph-based approximate nearest neighbor index.',
   '{"source": "indexes.md", "chunk": 0}', '[0.12, 0.22, 0.29]', 'nomic-embed-text:v1.5');

-- For very large loads (hundreds of thousands+ rows), COPY is the fastest path:
--   COPY documents (content, embedding, model_id) FROM '/path/file.csv' CSV;


-- ============================================================================
--  5. THE CORE RAG SEARCH QUERY  ("find the k nearest chunks")
-- ============================================================================
-- This is the query you run on EVERY user question. The pattern never changes:
--   1) embed the question in your app -> get a query vector
--   2) ORDER BY embedding <=> :query_vector   (nearest first)
--   3) LIMIT k                                  (top-k chunks to feed the LLM)
--
-- We also compute `1 - distance` AS similarity so the result is readable
-- (1.0 = identical, 0.0 = unrelated). The :query_vector is a placeholder your
-- driver fills in — NEVER string-concatenate it (SQL injection + slow).
SELECT
  id,
  content,
  metadata,
  1 - (embedding <=> '[0.1, 0.2, 0.3]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector   -- nearest neighbor first
LIMIT 5;                                            -- top-k (k = 5 here)

-- WHY ORDER BY the operator and not the `similarity` alias?
--   The index (section 7) can ONLY be used when you ORDER BY the raw distance
--   expression `embedding <=> :q`. Ordering by `1 - (...)` or by the alias
--   would prevent the index from being used. So: ORDER BY the distance,
--   SELECT the prettified similarity.


-- ============================================================================
--  6. METADATA FILTERING  (the most important real-world pattern)
-- ============================================================================
-- Pure vector search returns the globally nearest chunks. Usually you want
-- "nearest chunks WITHIN this user's documents" or "...from this source".
-- Combine a WHERE clause on metadata/columns with the vector ORDER BY.

-- Filter by a JSONB field (->> extracts a JSON value AS TEXT):
SELECT content, 1 - (embedding <=> '[0.1,0.2,0.3]'::vector) AS similarity
FROM documents
WHERE metadata->>'source' = 'intro.md'             -- only this source file
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 5;

-- Add a similarity THRESHOLD so you don't feed the LLM weak/irrelevant matches.
-- Here we keep only rows with cosine similarity >= 0.5 (distance <= 0.5).
SELECT content, 1 - (embedding <=> '[0.1,0.2,0.3]'::vector) AS similarity
FROM documents
WHERE (embedding <=> '[0.1,0.2,0.3]'::vector) <= 0.5
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 5;

-- To make metadata filters fast at scale, index the JSONB column:
CREATE INDEX IF NOT EXISTS idx_docs_metadata ON documents USING gin (metadata);

-- CAUTION — "post-filtering" with ANN indexes: an approximate index finds k
-- candidates FIRST, then your WHERE filters them. If the filter is very
-- selective you can get back FEWER than k rows (or zero) even when matches
-- exist. Mitigations: raise the search breadth (hnsw.ef_search, section 8),
-- or use partitioning / partial indexes for highly selective tenants.


-- ============================================================================
--  7. ANN INDEXES — making search fast (HNSW vs IVFFlat)
-- ============================================================================
-- Without an index, every query does an EXACT scan of ALL rows (brute force).
-- That's fine for a few thousand rows and is 100% accurate. Past that you want
-- an APPROXIMATE index: slightly less exact, dramatically faster.
--
-- pgvector offers two index types. The op-class in the index MUST match the
-- operator you query with:
--    vector_cosine_ops   <-> for the  <=>  operator   (cosine)
--    vector_ip_ops       <-> for the  <#>  operator   (inner product)
--    vector_l2_ops       <-> for the  <->  operator   (L2 / euclidean)
-- Mismatch = the index is silently ignored.

-- ---- HNSW (recommended default for RAG) ----------------------------------
-- Graph-based. Best query speed + recall; builds slower and uses more memory.
-- Great when data is added incrementally (you don't need existing rows first).
--    m               = edges per node. Higher = better recall, more memory/slower
--                      build. 16 is a solid default.
--    ef_construction = build-time search breadth. Higher = better index quality,
--                      slower build. 64 is a good default.
CREATE INDEX IF NOT EXISTS idx_docs_hnsw
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ---- IVFFlat (alternative) -----------------------------------------------
-- Clusters vectors into `lists` buckets; queries search the nearest buckets.
-- Builds faster & smaller than HNSW, but lower recall and — importantly —
-- you should build it AFTER the table already has representative data, because
-- the cluster centers are learned from existing rows.
--    lists = number of buckets. Rule of thumb: rows/1000 (up to ~1M rows),
--            then sqrt(rows) beyond that. Re-tune as the table grows.
-- CREATE INDEX idx_docs_ivfflat
-- ON documents
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- QUICK CHOICE: default to HNSW for RAG. Consider IVFFlat only if index build
-- time / memory is a real constraint and you can tolerate lower recall.


-- ============================================================================
--  8. QUERY-TIME RECALL/SPEED TUNING
-- ============================================================================
-- Each index type has a runtime knob that trades recall for speed. Set it per
-- session (or per transaction) BEFORE running your search.

-- HNSW: how many candidates to explore. Higher = better recall, slower.
-- Must be >= your LIMIT. Default is 40. Bump it if you're missing good matches.
SET hnsw.ef_search = 100;

-- IVFFlat: how many buckets to probe. Higher = better recall, slower.
-- Default is 1; for decent recall try lists/10-ish.
SET ivfflat.probes = 10;

-- Tip: SET LOCAL inside a transaction scopes the change to that transaction
-- only, which is cleaner than changing it globally for the whole connection.


-- ============================================================================
--  9. VERIFY THE INDEX IS ACTUALLY USED  (don't assume — measure)
-- ============================================================================
-- EXPLAIN ANALYZE shows the real plan + timing. You want to see an
-- "Index Scan using idx_docs_hnsw", NOT a "Seq Scan".
EXPLAIN ANALYZE
SELECT content, embedding <=> '[0.1,0.2,0.3]'::vector AS distance
FROM documents
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 5;
-- COMMON REASONS YOU SEE A "Seq Scan" INSTEAD OF THE INDEX:
--   * Too few rows — the planner correctly decides brute force is cheaper.
--     (For ~7 rows, a seq scan genuinely IS faster than walking a graph.)
--   * Dimension mismatch — query vector dims != column dims. The index can't
--     be used (and real execution would error). Always query with the same
--     dimension you stored.
--   * Operator/op-class mismatch — e.g. index built with vector_cosine_ops but
--     you queried with <-> (L2). They must match.
--   * You ORDER BY a wrapped expression (1 - (...)) instead of the raw <=>.


-- ============================================================================
--  10. MAINTENANCE & USEFUL INSPECTION QUERIES
-- ============================================================================
-- How big is the table / are vectors there?
SELECT count(*) AS total_chunks,
       count(embedding) AS chunks_with_embeddings
FROM documents;

-- See your indexes:
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'documents';

-- Table & index sizes on disk (vectors are big — watch this grow):
SELECT pg_size_pretty(pg_total_relation_size('documents')) AS total_size;

-- After large data loads/changes, refresh planner statistics so it makes good
-- decisions about when to use the index:
ANALYZE documents;

-- Updating a chunk's embedding (e.g. after re-embedding with a new model):
UPDATE documents
SET embedding = '[0.15, 0.25, 0.35]', model_id = 'text-embedding-3-small'
WHERE id = '00000000-0000-0000-0000-000000000000';

-- Find rows still on an OLD model so you know what to re-embed after a switch:
SELECT id, content FROM documents WHERE model_id <> 'nomic-embed-text:v1.5';

-- Delete a document's chunks (e.g. when the source file is removed):
DELETE FROM documents WHERE metadata->>'source' = 'intro.md';


-- ============================================================================
--  11. THINGS THAT BITE PEOPLE (read before going to production)
-- ============================================================================
-- * DIMENSIONS ARE FOREVER (per model). vector(768) can't hold a 1536-dim
--   vector. Changing embedding models = a migration: re-embed every row.
--
-- * NEVER MIX MODELS in one similarity search. Vectors from different models
--   live in different spaces; distances between them are meaningless. That's
--   why we store model_id.
--
-- * NORMALIZE if your model expects it. For cosine search this doesn't matter
--   (cosine ignores magnitude), but for inner-product (<#>) search on
--   un-normalized vectors, results can be wrong. When in doubt, use cosine.
--
-- * >2000 DIMENSIONS need `halfvec` for indexing. A regular `vector` index is
--   capped at 2000 dims. For 3072-dim models (text-embedding-3-large), store as
--   halfvec(3072) (half-precision, half the storage) and index with
--   halfvec_cosine_ops, or reduce dimensions at embed time if the model allows.
--
-- * APPROXIMATE MEANS APPROXIMATE. ANN indexes can miss the true nearest
--   neighbor. Tune ef_search / probes for the recall you need, and remember a
--   brute-force (no-index) scan is the 100%-accurate ground truth to compare
--   against while you tune.
--
-- * keep the raw `content`. The vector finds the chunk; the LLM needs the
--   text. Losing the text means a useless retrieval system.
--
-- * BACK UP like any Postgres DB (pg_dump). Your embeddings are expensive to
--   recompute — treat them as valuable data, not a disposable cache.
-- ============================================================================
