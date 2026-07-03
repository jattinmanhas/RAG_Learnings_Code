// db.go — Postgres + pgvector plumbing.
//
// We deliberately pass vectors to Postgres as their text literal form,
// '[0.1,0.2,...]', and cast with $N::vector in SQL. That's exactly what the
// TypeScript example does and it keeps the dependency surface tiny — no
// pgvector-specific driver package needed. For very high write throughput you'd
// switch to the binary protocol (github.com/pgvector/pgvector-go), but the
// semantics are identical.
package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

func databaseURL() string {
	if u := os.Getenv("DATABASE_URL"); u != "" {
		return u
	}
	// Matches docker-compose.yml in the parent folder.
	return "postgres://user:password@localhost:5432/vectordb"
}

func newPool(ctx context.Context) (*pgxpool.Pool, error) {
	return pgxpool.New(ctx, databaseURL())
}

// toVector renders an embedding as the pgvector text literal '[x,y,z]'.
// strconv with -1 precision emits the shortest round-trippable form, avoiding
// both precision loss and needlessly long strings.
func toVector(embedding []float32) string {
	var b strings.Builder
	b.Grow(len(embedding) * 8)
	b.WriteByte('[')
	for i, v := range embedding {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(v), 'f', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

// ensureSchema creates the extension, table, and HNSW index if they don't yet
// exist, so the demo is self-contained (the TS example assumes you ran
// src/pgsql.sql by hand). Idempotent — safe to run every startup.
func ensureSchema(ctx context.Context, pool *pgxpool.Pool) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE TABLE IF NOT EXISTS documents (
			id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			content    TEXT NOT NULL,
			metadata   JSONB DEFAULT '{}',
			embedding  vector(768),
			model_id   TEXT NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		// vector_cosine_ops → the <=> (cosine distance) operator uses this index.
		// Each distance metric needs its OWN opclass/index; see search.go.
		`CREATE INDEX IF NOT EXISTS idx_docs_embedding_hnsw
			ON documents USING hnsw (embedding vector_cosine_ops)
			WITH (m = 16, ef_construction = 64)`,
		// A GIN index makes metadata filters (WHERE metadata @> ...) fast, which
		// matters for the hybrid filtered-search example.
		`CREATE INDEX IF NOT EXISTS idx_docs_metadata ON documents USING gin (metadata)`,
	}
	for _, s := range stmts {
		if _, err := pool.Exec(ctx, s); err != nil {
			return fmt.Errorf("schema stmt failed: %w\n%s", err, s)
		}
	}
	return nil
}

// insertDocument embeds content and stores it with optional JSON metadata.
func insertDocument(ctx context.Context, pool *pgxpool.Pool, content string, metadata map[string]any) (string, error) {
	embedding, err := getEmbedding(ctx, content)
	if err != nil {
		return "", err
	}

	var id string
	err = pool.QueryRow(ctx,
		`INSERT INTO documents (content, embedding, metadata, model_id)
		 VALUES ($1, $2::vector, $3, $4)
		 RETURNING id`,
		content, toVector(embedding), metadata, embeddingModel,
	).Scan(&id)
	return id, err
}
