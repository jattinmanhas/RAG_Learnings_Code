CREATE TABLE documents (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content    TEXT NOT NULL,
  metadata   JSONB DEFAULT '{}',
  embedding  vector(768),
  model_id   TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_docs_embedding_hnsw
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

/*
vectordb=# \d documents
                             Table "public.documents"
   Column   |           Type           | Collation | Nullable |      Default      
------------+--------------------------+-----------+----------+-------------------
 id         | uuid                     |           | not null | gen_random_uuid()
 content    | text                     |           | not null | 
 metadata   | jsonb                    |           |          | '{}'::jsonb
 embedding  | vector(768)              |           |          | 
 model_id   | text                     |           | not null | 
 created_at | timestamp with time zone |           |          | now()
Indexes:
    "documents_pkey" PRIMARY KEY, btree (id)
    "idx_docs_embedding_hnsw" hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')

=========================================

vectordb=# EXPLAIN
SELECT id, 1 - (embedding <=> '[0.1,0.2,0.3]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 5;
                               QUERY PLAN                                
-------------------------------------------------------------------------
 Limit  (cost=25.35..25.36 rows=5 width=32)
   ->  Sort  (cost=25.35..26.47 rows=450 width=32)
         Sort Key: ((embedding <=> '[0.1,0.2,0.3]'::vector))
         ->  Seq Scan on documents  (cost=0.00..17.88 rows=450 width=32)
(4 rows)

*/