// main.go — runnable demo of pgvector-backed vector search in Go.
//
//	cd "2.2 Vector Search/go"
//	go mod tidy
//	docker compose -f ../docker-compose.yml up -d   # Postgres + pgvector
//	ollama pull nomic-embed-text:v1.5               # 768-dim embeddings
//	go run .
//
// Override defaults with env vars:
//
//	DATABASE_URL=postgres://user:password@localhost:5432/vectordb
//	OLLAMA_URL=http://localhost:11434
//
// The demo is idempotent: it creates the schema if needed and seeds rows only
// when the table is empty, so you can re-run it freely.
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	pool, err := newPool(ctx)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	if err := ensureSchema(ctx, pool); err != nil {
		log.Fatalf("schema: %v", err)
	}
	if err := seed(ctx, pool); err != nil {
		log.Fatalf("seed: %v", err)
	}

	// 1. Basic semantic search — same query as the TypeScript example.
	fmt.Println("── 1. Semantic search ─────────────────────────────")
	results, err := semanticSearch(ctx, pool, "vector search in databases", 3)
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	printResults(results)

	// 2. Distance metrics side by side.
	fmt.Println("\n── 2. Distance metrics ────────────────────────────")
	if err := compareDistanceMetrics(ctx, pool, "vector search in databases"); err != nil {
		log.Fatalf("metrics: %v", err)
	}

	// 3. Hybrid search: vector similarity constrained by a metadata filter.
	//    Only rows tagged {"category":"database"} are eligible, then ranked by
	//    similarity — note "in-memory key-value store" (Redis) still wins within
	//    that category even though it's not the closest doc overall.
	fmt.Println("\n── 3. Filtered (hybrid) search ────────────────────")
	filtered, err := filteredSearch(ctx, pool, "fast lookups",
		map[string]any{"category": "database"}, 3)
	if err != nil {
		log.Fatalf("filtered: %v", err)
	}
	printResults(filtered)

	// 4. ef_search recall/speed tradeoff, demonstrated live.
	fmt.Println("\n── 4. ef_search recall vs speed ───────────────────")
	if err := demoEfSearch(ctx, pool, "distributed systems and streaming"); err != nil {
		log.Fatalf("ef_search: %v", err)
	}

	// 5. What the planner actually does.
	fmt.Println("\n── 5. Query plan (index vs seq scan) ──────────────")
	if err := explainSearch(ctx, pool, "vector search in databases"); err != nil {
		log.Fatalf("explain: %v", err)
	}
}

// demoEfSearch runs the same nearest-neighbor query at several ef_search values
// and times each. At small scale the numbers are noisy, but the shape is the
// lesson: raising ef_search widens the graph search → higher recall, more work.
func demoEfSearch(ctx context.Context, pool *pgxpool.Pool, query string) error {
	embedding, err := getEmbedding(ctx, query)
	if err != nil {
		return err
	}
	vec := toVector(embedding)

	for _, ef := range []int{10, 40, 100} {
		conn, err := pool.Acquire(ctx)
		if err != nil {
			return err
		}
		if _, err := conn.Exec(ctx, fmt.Sprintf("SET hnsw.ef_search = %d", ef)); err != nil {
			conn.Release()
			return err
		}
		start := time.Now()
		var topContent string
		err = conn.QueryRow(ctx,
			`SELECT content FROM documents ORDER BY embedding <=> $1::vector LIMIT 1`,
			vec,
		).Scan(&topContent)
		elapsed := time.Since(start)
		conn.Release()
		if err != nil {
			return err
		}
		fmt.Printf("  ef_search=%-3d  %-8v  top: %s\n", ef, elapsed.Round(time.Microsecond), topContent)
	}
	return nil
}

func printResults(results []Result) {
	if len(results) == 0 {
		fmt.Println("  (no results)")
		return
	}
	for i, r := range results {
		fmt.Printf("  %d. [%.4f] %s\n", i+1, r.Similarity, r.Content)
	}
}

// seed inserts a small corpus the first time the table is empty. Each doc
// carries a category in metadata so the filtered-search example has something
// to constrain on.
func seed(ctx context.Context, pool *pgxpool.Pool) error {
	var count int
	if err := pool.QueryRow(ctx, `SELECT COUNT(*) FROM documents`).Scan(&count); err != nil {
		return err
	}
	if count > 0 {
		fmt.Printf("Corpus already has %d documents, skipping seed.\n\n", count)
		return nil
	}

	docs := []struct {
		content  string
		category string
	}{
		{"Postgres is a powerful open-source relational database.", "database"},
		{"Redis is an in-memory key-value store used for caching.", "database"},
		{"Kafka is a distributed event streaming platform.", "infra"},
		{"Docker makes it easy to run applications in containers.", "infra"},
		{"pgvector adds vector similarity search to Postgres.", "database"},
		{"Embeddings are dense numerical representations of text.", "ml"},
		{"HNSW is a graph-based approximate nearest neighbor algorithm.", "ml"},
	}

	fmt.Println("Seeding corpus...")
	for _, d := range docs {
		id, err := insertDocument(ctx, pool, d.content, map[string]any{"category": d.category})
		if err != nil {
			return err
		}
		fmt.Printf("  inserted %s  (%s)\n", id[:8], d.content[:min(40, len(d.content))])
	}
	fmt.Println()
	return nil
}
