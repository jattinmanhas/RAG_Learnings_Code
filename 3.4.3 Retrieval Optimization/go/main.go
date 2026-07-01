// main.go — runnable demo for Retrieval Optimization.
//
//	go run .
//
// No API key required. mockEmbed turns text into a 6-dimensional concept vector
// via keyword scoring — enough to show dense retrieval, and to contrast it with
// BM25 keyword retrieval, without a network call. Replace mockEmbed with a real
// provider (Voyage AI, Cohere, OpenAI) in production.
//
// The demo walks the four levers in order:
//   1. Top-K retrieval        — dense baseline, varying K.
//   2. Metadata filtering      — same query, restricted by structured fields.
//   3. Hybrid retrieval        — dense vs sparse vs fused on an exact-token query.
//   4. Retrieval tuning        — sweeping alpha, RRF vs weighted, thresholds.
package main

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// ---------------------------------------------------------------------------
// SAMPLE CORPUS
// ---------------------------------------------------------------------------
// A small knowledge base about vector databases. Each chunk carries metadata
// you'd realistically filter on: source, language, and access level. Note the
// exact tokens ("HNSW", "error code E1401", "pgvector 0.7") — those are where
// dense-only retrieval struggles and BM25 shines.

var corpus = []Chunk{
	{
		ID:   "hnsw-overview",
		Text: "HNSW is a graph-based approximate nearest neighbour index. It builds a multi-layer proximity graph and offers high recall with real-time inserts.",
		Metadata: map[string]string{
			"source": "docs", "language": "en", "access": "public",
		},
	},
	{
		ID:   "ivfflat-overview",
		Text: "IVFFlat partitions the vector space into Voronoi cells with k-means. It uses less memory than HNSW but must be rebuilt when the cluster count changes.",
		Metadata: map[string]string{
			"source": "docs", "language": "en", "access": "public",
		},
	},
	{
		ID:   "pgvector-release",
		Text: "pgvector 0.7 adds halfvec and sparse vector support, improving memory usage for large embedding tables inside Postgres.",
		Metadata: map[string]string{
			"source": "changelog", "language": "en", "access": "public",
		},
	},
	{
		ID:   "error-e1401",
		Text: "Troubleshooting error code E1401: the embedding dimension of the query does not match the index dimension. Re-embed with the correct model.",
		Metadata: map[string]string{
			"source": "support", "language": "en", "access": "internal",
		},
	},
	{
		ID:   "recall-tuning",
		Text: "To raise recall, increase the number of candidates retrieved before re-ranking, or widen the search parameter so the graph traversal visits more nodes.",
		Metadata: map[string]string{
			"source": "docs", "language": "en", "access": "public",
		},
	},
	{
		ID:   "memoria-vectorial",
		Text: "La memoria vectorial almacena embeddings densos para la busqueda semantica. HNSW ofrece alta exhaustividad con inserciones en tiempo real.",
		Metadata: map[string]string{
			"source": "docs", "language": "es", "access": "public",
		},
	},
}

// ---------------------------------------------------------------------------
// MOCK EMBEDDING FUNCTION
// ---------------------------------------------------------------------------
// Maps text to a 6-dimensional concept vector by counting keyword matches per
// semantic dimension. Deliberately blind to exact tokens like "E1401" or
// "pgvector 0.7" — that blindness is precisely why the hybrid demo needs BM25.
//
// Dimension legend:
//   [0] graph-index / HNSW concepts
//   [1] partition / IVFFlat / clustering concepts
//   [2] memory / storage concepts
//   [3] recall / tuning concepts
//   [4] error / troubleshooting concepts
//   [5] Postgres / pgvector concepts

var conceptKeywords = [][]string{
	{"hnsw", "graph", "proximity", "layer", "navigable", "node", "traversal", "exhaustividad", "vectorial"},
	{"ivfflat", "ivf", "voronoi", "cluster", "kmeans", "k-means", "partition", "centroid"},
	{"memory", "storage", "halfvec", "usage", "memoria", "densos"},
	{"recall", "candidate", "rerank", "re-ranking", "tuning", "search", "widen", "raise"},
	{"error", "troubleshoot", "troubleshooting", "mismatch", "dimension", "match"},
	{"pgvector", "postgres", "sparse", "table", "embedding"},
}

func mockEmbed(_ context.Context, text string) ([]float64, error) {
	lower := strings.ToLower(text)
	vec := make([]float64, len(conceptKeywords))
	for dim, kws := range conceptKeywords {
		for _, kw := range kws {
			vec[dim] += float64(strings.Count(lower, kw))
		}
	}
	return l2Normalize(vec), nil
}

func l2Normalize(v []float64) []float64 {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	if norm == 0 {
		return v
	}
	norm = math.Sqrt(norm)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// ---------------------------------------------------------------------------
// DISPLAY HELPERS
// ---------------------------------------------------------------------------

const hr = "============================================================================"

func header(title string) { fmt.Printf("\n%s\n%s\n%s\n", hr, title, hr) }

func truncateText(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func printResults(results []Scored, showComponents bool) {
	if len(results) == 0 {
		fmt.Println("  (no results)")
		return
	}
	for i, r := range results {
		fmt.Printf("  [%d] %-18s score=%.4f", i+1, r.Chunk.ID, r.Score)
		if showComponents {
			fmt.Printf("  (dense=%.3f#%d  sparse=%.3f#%d)",
				r.DenseScore, r.DenseRank, r.SparseScore, r.SparseRank)
		}
		fmt.Printf("\n        %q\n", truncateText(r.Chunk.Text, 78))
	}
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

func main() {
	ctx := context.Background()
	store := NewStore(corpus, defaultBM25())

	// =======================================================================
	// TOPIC 1 — TOP-K RETRIEVAL
	// =======================================================================
	header("TOPIC 1 — Top-K Retrieval (dense only)")

	q1 := "How does HNSW indexing work and when should I use it?"
	fmt.Printf("  Query: %q\n", q1)

	for _, k := range []int{2, 4} {
		opts := DefaultOptions()
		opts.Hybrid = false // pure dense to isolate the K knob
		opts.TopK = k
		res, err := store.Retrieve(ctx, q1, mockEmbed, opts)
		must(err)
		fmt.Printf("\n  TopK = %d:\n", k)
		printResults(res, false)
	}
	fmt.Println("\n  → K trades recall for precision/latency. Too small and you miss a")
	fmt.Println("    relevant chunk; too large and you pad the prompt with noise.")

	// =======================================================================
	// TOPIC 2 — METADATA FILTERING
	// =======================================================================
	header("TOPIC 2 — Metadata Filtering")

	q2 := "vector memory for semantic search"
	fmt.Printf("  Query: %q\n", q2)

	base := DefaultOptions()
	base.Hybrid = false

	fmt.Println("\n  No filter (all languages, all access levels):")
	res, err := store.Retrieve(ctx, q2, mockEmbed, base)
	must(err)
	printResults(res, false)

	fmt.Println("\n  Filter: language == \"en\"  (drop the Spanish chunk):")
	optEn := base
	optEn.Filter = FieldEquals("language", "en")
	res, err = store.Retrieve(ctx, q2, mockEmbed, optEn)
	must(err)
	printResults(res, false)

	fmt.Println("\n  Filter: language == \"en\" AND access == \"public\"  (security filter):")
	optSecure := base
	optSecure.Filter = And(
		FieldEquals("language", "en"),
		FieldIn("access", "public"),
	)
	res, err = store.Retrieve(ctx, q2, mockEmbed, optSecure)
	must(err)
	printResults(res, false)
	fmt.Println("\n  → Filtering runs BEFORE scoring. The 'internal' error-E1401 chunk can")
	fmt.Println("    never leak to a public user, regardless of how well it matches.")

	// =======================================================================
	// TOPIC 3 — HYBRID RETRIEVAL
	// =======================================================================
	header("TOPIC 3 — Hybrid Retrieval (dense + BM25)")

	// An exact-token lookup: the dense embedder has never seen the token
	// "E1401", so it can't tell the chunks apart — but BM25 nails it. Hybrid
	// fusion recovers the exact match the vectors miss.
	q3 := "look up the E1401 issue"
	fmt.Printf("  Query: %q\n", q3)
	fmt.Println("  (note: the mock embedder has no concept for the exact token 'E1401')")

	optDense := DefaultOptions()
	optDense.Hybrid = false
	optDense.Filter = MatchAll
	optDense.TopK = 3
	fmt.Println("\n  Dense only — misses the exact match:")
	res, err = store.Retrieve(ctx, q3, mockEmbed, optDense)
	must(err)
	printResults(res, true)

	optHybrid := DefaultOptions()
	optHybrid.Filter = MatchAll
	optHybrid.TopK = 3
	optHybrid.Fusion = FusionRRF
	fmt.Println("\n  Hybrid (RRF fusion) — BM25 surfaces the exact match, fused to the top:")
	res, err = store.Retrieve(ctx, q3, mockEmbed, optHybrid)
	must(err)
	printResults(res, true)
	fmt.Println("\n  → Dense gives semantic recall; BM25 gives exact-token precision.")
	fmt.Println("    RRF fuses the two rankings without needing to normalise scores.")

	// =======================================================================
	// TOPIC 4 — RETRIEVAL TUNING
	// =======================================================================
	header("TOPIC 4 — Retrieval Tuning (sweeping the knobs)")

	q4 := "how to raise recall with HNSW"
	fmt.Printf("  Query: %q\n", q4)

	fmt.Println("\n  (a) Weighted fusion, sweeping alpha (dense weight):")
	for _, alpha := range []float64{0.0, 0.5, 1.0} {
		o := DefaultOptions()
		o.Filter = MatchAll
		o.TopK = 3
		o.Fusion = FusionWeighted
		o.Alpha = alpha
		res, err := store.Retrieve(ctx, q4, mockEmbed, o)
		must(err)
		label := fmt.Sprintf("alpha=%.1f", alpha)
		switch alpha {
		case 0.0:
			label += " (pure BM25)"
		case 1.0:
			label += " (pure dense)"
		}
		fmt.Printf("\n    %s:\n", label)
		printResults(res, true)
	}

	fmt.Println("\n  (b) RRF constant — smaller rrfK sharpens top-rank dominance:")
	for _, rrfK := range []float64{5, 60} {
		o := DefaultOptions()
		o.Filter = MatchAll
		o.TopK = 3
		o.Fusion = FusionRRF
		o.RRFK = rrfK
		res, err := store.Retrieve(ctx, q4, mockEmbed, o)
		must(err)
		fmt.Printf("\n    rrfK=%.0f:\n", rrfK)
		printResults(res, true)
	}

	fmt.Println("\n  (c) Score threshold — drop weak matches (abstention signal):")
	o := DefaultOptions()
	o.Filter = MatchAll
	o.TopK = 5
	o.Fusion = FusionRRF
	o.Threshold = 0.0322
	res, err = store.Retrieve(ctx, q4, mockEmbed, o)
	must(err)
	fmt.Printf("    Threshold=%.4f keeps %d of up to 5 (weak matches dropped):\n", o.Threshold, len(res))
	printResults(res, true)

	fmt.Printf("\n%s\n", hr)
	fmt.Println("Done. In production:")
	fmt.Println("  • Replace mockEmbed with a real provider (Voyage AI, Cohere, OpenAI).")
	fmt.Println("  • Replace the in-memory dense search with a pgvector ORDER BY <=> query.")
	fmt.Println("  • Replace BM25 with Postgres full-text (tsvector) or Elasticsearch.")
	fmt.Println("  • Push the metadata Filter into the SQL WHERE clause so the index does it.")
	fmt.Println("  • Tune TopK / alpha / rrfK / Threshold against a labelled eval set (3.10).")
	fmt.Printf("%s\n", hr)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
