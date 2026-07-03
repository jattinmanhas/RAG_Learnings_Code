// search.go — the vector-search operations worth understanding, each isolated
// so you can read them one at a time.
//
// pgvector distance operators (lower = closer for all of them):
//
//	<=>  cosine distance          → similarity = 1 - (a <=> b)
//	<->  Euclidean / L2 distance  → straight-line distance
//	<#>  negative inner product   → returns -(a · b), so more-negative = closer
//
// Which to pick? Match the operator to how your embedding model was trained.
// nomic-embed-text returns normalized vectors, so cosine and inner product rank
// identically; cosine is the safe default and what our HNSW index is built for.
package main

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Result struct {
	ID         string
	Content    string
	Similarity float64
}

// semanticSearch is the bread-and-butter query: nearest neighbors by cosine
// distance. This mirrors the TypeScript example exactly.
func semanticSearch(ctx context.Context, pool *pgxpool.Pool, query string, limit int) ([]Result, error) {
	embedding, err := getEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}
	vec := toVector(embedding)

	conn, err := pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer conn.Release()

	// ef_search is the query-time recall/speed dial for HNSW: it bounds how many
	// candidate nodes the graph walk keeps. Higher = better recall, slower.
	// Default 40. It's a per-session GUC, so set it on the pooled connection we
	// actually run the query on.
	if _, err := conn.Exec(ctx, "SET hnsw.ef_search = 64"); err != nil {
		return nil, err
	}

	rows, err := conn.Query(ctx,
		`SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
		 FROM documents
		 ORDER BY embedding <=> $1::vector
		 LIMIT $2`,
		vec, limit,
	)
	if err != nil {
		return nil, err
	}
	return collect(rows)
}

// filteredSearch shows hybrid retrieval: a structured metadata predicate AND
// vector similarity in one query. This is how you scope RAG to a tenant, a
// document set, a language, a date range, etc. Postgres evaluates the WHERE
// filter and the ORDER BY vector distance together — the planner decides
// whether to filter-then-rank or use the vector index and filter after, based
// on selectivity. The @> containment operator is backed by the GIN index.
func filteredSearch(ctx context.Context, pool *pgxpool.Pool, query string, filter map[string]any, limit int) ([]Result, error) {
	embedding, err := getEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}

	rows, err := pool.Query(ctx,
		`SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
		 FROM documents
		 WHERE metadata @> $2
		 ORDER BY embedding <=> $1::vector
		 LIMIT $3`,
		toVector(embedding), filter, limit,
	)
	if err != nil {
		return nil, err
	}
	return collect(rows)
}

// compareDistanceMetrics runs the SAME query vector through all three operators
// and prints the top hit for each. On normalized embeddings cosine and inner
// product agree; L2 usually agrees too but can diverge on magnitude-sensitive
// data. Seeing them side by side makes the abstract operators concrete.
func compareDistanceMetrics(ctx context.Context, pool *pgxpool.Pool, query string) error {
	embedding, err := getEmbedding(ctx, query)
	if err != nil {
		return err
	}
	vec := toVector(embedding)

	metrics := []struct {
		name string
		expr string // distance expression; lower = closer
	}{
		{"cosine   (<=>)", "embedding <=> $1::vector"},
		{"L2       (<->)", "embedding <-> $1::vector"},
		{"innerprod(<#>)", "embedding <#> $1::vector"},
	}

	fmt.Printf("Top match per distance metric for %q:\n", query)
	for _, m := range metrics {
		var content string
		var distance float64
		err := pool.QueryRow(ctx,
			fmt.Sprintf(`SELECT content, %s AS distance
			             FROM documents ORDER BY %s LIMIT 1`, m.expr, m.expr),
			vec,
		).Scan(&content, &distance)
		if err != nil {
			return err
		}
		fmt.Printf("  %s  dist=%+.4f  %s\n", m.name, distance, content)
	}
	return nil
}

// explainSearch prints the planner's chosen path. With only a handful of rows
// you'll see a Seq Scan — brute force genuinely beats walking an HNSW graph at
// tiny scale, and that's correct, not a bug. The index earns its keep once the
// table grows into the hundreds/thousands of rows. Compare against forcing the
// index with `SET enable_seqscan = off` inside psql.
func explainSearch(ctx context.Context, pool *pgxpool.Pool, query string) error {
	embedding, err := getEmbedding(ctx, query)
	if err != nil {
		return err
	}

	rows, err := pool.Query(ctx,
		`EXPLAIN
		 SELECT id, content FROM documents
		 ORDER BY embedding <=> $1::vector
		 LIMIT 5`,
		toVector(embedding),
	)
	if err != nil {
		return err
	}
	defer rows.Close()

	fmt.Println("EXPLAIN plan:")
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			return err
		}
		fmt.Printf("  %s\n", line)
	}
	return rows.Err()
}

func collect(rows pgx.Rows) ([]Result, error) {
	defer rows.Close()
	var out []Result
	for rows.Next() {
		var r Result
		if err := rows.Scan(&r.ID, &r.Content, &r.Similarity); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
