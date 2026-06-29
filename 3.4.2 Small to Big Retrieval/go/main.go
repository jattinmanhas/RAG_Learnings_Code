// main.go — runnable demo for Small-to-Big (Sentence-Window) Retrieval.
//
//	go run .
//
// No API key is required.  The mock embedding function converts text into a
// 6-dimensional concept vector using keyword scoring — good enough to show
// the full pipeline without a network call.  Replace mockEmbed with a real
// provider call in production.
package main

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// ---------------------------------------------------------------------------
// SAMPLE DOCUMENTS
// ---------------------------------------------------------------------------
// Unlike parent-child (3.4.1), small-to-big does NOT pre-split documents into
// sections.  Each document is just continuous prose; IngestDocument breaks it
// into individual SENTENCES, and the window expansion at query time decides
// how much surrounding context to pull back.
//
// We use two documents so you can see that window expansion is bounded to the
// same document (no cross-document bleed).

var documents = map[string]string{
	"indexing": `HNSW is the most widely deployed approximate nearest-neighbour index for dense vector search. It builds a multi-layer proximity graph where each node is a vector. Queries traverse the graph from a random entry point at the top layer and greedily descend to denser layers. HNSW achieves sub-linear query time and can be updated incrementally without a rebuild. IVFFlat instead partitions the vector space into Voronoi cells using k-means clustering. At query time only the nprobe closest centroids are searched, reducing the comparison set dramatically. Unlike HNSW, IVFFlat requires a full rebuild when nlist changes and cannot be updated incrementally. HNSW consumes more memory but supports real-time inserts and higher recall. IVFFlat uses less memory and is faster to build on a large static corpus. For most production RAG workloads with ongoing ingestion, HNSW is the safer default.`,

	"chunking": `Fixed-size chunking divides a document into non-overlapping windows of N tokens. It is the simplest strategy to implement and produces uniform index sizes. The main risk is boundary cuts that split a sentence mid-clause and lose meaning. Semantic chunking instead groups sentences by measuring the embedding distance between consecutive sentences. When the distance spikes above a threshold, a new chunk begins. This keeps topically coherent passages together at the cost of variable chunk sizes. Overlap is a complementary technique where a sliding window shares tokens between adjacent chunks. Overlap ensures information near a boundary appears in at least two chunks. A twenty percent overlap is a common starting point but increases index size proportionally.`,
}

// docOrder fixes a stable iteration order for printing (Go maps are unordered).
var docOrder = []string{"indexing", "chunking"}

// ---------------------------------------------------------------------------
// MOCK EMBEDDING FUNCTION
// ---------------------------------------------------------------------------
// Maps any text to a 6-dimensional "concept vector" by counting keyword
// matches across six semantic dimensions.  Each dimension corresponds to a
// distinct topic cluster across the two sample documents above.
//
// Dimension legend:
//   [0] HNSW / graph-index concepts
//   [1] IVFFlat / centroid / Voronoi concepts
//   [2] memory / recall / build trade-off concepts
//   [3] fixed-size / boundary chunking concepts
//   [4] semantic chunking / distance-threshold concepts
//   [5] overlap / sliding window concepts
//
// The resulting vector is L2-normalised so cosine similarity equals the dot
// product.  In production, replace this with a real API call.
//
// In production use:
//   import "github.com/anthropics/anthropic-sdk-go"
//   // Anthropic doesn't provide embeddings directly; use via a compatible
//   // embedding provider (Voyage AI, Cohere, OpenAI) and plug it in here.
//
//   Or with Voyage AI (recommended alongside Claude):
//   resp, _ := voyageClient.Embed(ctx, &voyage.EmbedRequest{
//       Input: []string{text},
//       Model: "voyage-3",
//   })
//   return resp.Data[0].Embedding, nil

var conceptKeywords = [][]string{
	{"hnsw", "graph", "navigable", "layer", "node", "greedy", "descend", "entry", "proximity"},
	{"ivfflat", "ivf", "voronoi", "centroid", "cluster", "nprobe", "nlist", "kmeans", "k-means"},
	{"memory", "recall", "rebuild", "incremental", "insert", "build", "corpus", "ram"},
	{"fixed", "boundary", "uniform", "cut", "clause", "non-overlapping", "token"},
	{"semantic", "distance", "threshold", "coherent", "consecutive", "spike", "topically"},
	{"overlap", "stride", "sliding", "adjacent", "shared", "twenty", "percent", "proportional"},
}

func mockEmbed(_ context.Context, text string) ([]float64, error) {
	lower := strings.ToLower(text)
	vec := make([]float64, len(conceptKeywords))
	for dim, keywords := range conceptKeywords {
		for _, kw := range keywords {
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
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
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

func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "…"
}

// printStore prints a summary of what was ingested.
func printStore(store *SentenceStore) {
	fmt.Printf("\n  Ingested %d documents, %d sentences total.\n\n",
		len(store.byDoc), len(store.sentences))
	for _, docID := range docOrder {
		fmt.Printf("  doc %q — %d sentences:\n", docID, len(store.byDoc[docID]))
		for _, s := range store.byDoc[docID] {
			fmt.Printf("    [%d] %s  %q\n", s.Pos, s.ID, truncate(s.Text, 64))
		}
		fmt.Println()
	}
}

// printHits prints the raw sentence hits before window expansion.
func printHits(hits []SentenceScore) {
	for i, h := range hits {
		fmt.Printf("    [%d] score=%.4f  %s (doc=%s pos=%d)\n",
			i, h.Score, h.Sentence.ID, h.Sentence.DocID, h.Sentence.Pos)
		fmt.Printf("         %q\n", truncate(h.Sentence.Text, 70))
	}
}

// printWindows prints the merged "big" windows fed to the LLM.
func printWindows(results []WindowResult) {
	if len(results) == 0 {
		fmt.Println("  (no results)")
		return
	}
	for i, r := range results {
		fmt.Printf("  [%d] doc=%s  positions %d..%d  (%d sentences)\n",
			i, r.DocID, r.StartPos, r.EndPos, r.EndPos-r.StartPos+1)
		fmt.Printf("       bestScore = %.4f\n", r.BestScore)
		fmt.Printf("       matched   = %v\n", r.MatchedSentIDs)
		fmt.Printf("       context   → %q\n\n", truncate(r.Text, 160))
	}
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

func main() {
	ctx := context.Background()

	// -----------------------------------------------------------------------
	// INGESTION PHASE
	// -----------------------------------------------------------------------
	header("INGESTION PHASE — split documents into sentences")

	store := NewStore()
	for _, docID := range docOrder {
		IngestDocument(store, docID, documents[docID])
	}
	printStore(store)

	// -----------------------------------------------------------------------
	// QUERY 1 — targets the "indexing" document
	// -----------------------------------------------------------------------
	header("QUERY 1 — HNSW vs IVFFlat trade-offs (windowSize = 1)")

	q1 := "How does HNSW differ from IVFFlat and which uses more memory?"
	fmt.Printf("  Query: %q\n\n", q1)

	hits1, err := searchSentences(ctx, q1, store, mockEmbed, 4)
	must(err)
	fmt.Println("  Step 1 — Top-4 sentence hits (the 'small' chunks):")
	printHits(hits1)

	fmt.Println("\n  Step 2 — Expand ±1 sentence and merge overlaps (the 'big' chunks):")
	results1, err := Retrieve(ctx, store, q1, mockEmbed, 4, 1, 0)
	must(err)
	printWindows(results1)

	// -----------------------------------------------------------------------
	// QUERY 2 — show how a larger window pulls more context
	// -----------------------------------------------------------------------
	header("QUERY 2 — Same query, windowSize = 2 (wider context)")

	fmt.Printf("  Query: %q\n\n", q1)
	results2, err := Retrieve(ctx, store, q1, mockEmbed, 4, 2, 0)
	must(err)
	fmt.Println("  Wider windows → fewer, larger merged blocks:")
	printWindows(results2)

	// -----------------------------------------------------------------------
	// QUERY 3 — targets the "chunking" document
	// -----------------------------------------------------------------------
	header("QUERY 3 — Chunking overlap (windowSize = 1)")

	q3 := "What is overlap in chunking and how does it affect index size?"
	fmt.Printf("  Query: %q\n\n", q3)

	hits3, err := searchSentences(ctx, q3, store, mockEmbed, 3)
	must(err)
	fmt.Println("  Step 1 — Top-3 sentence hits:")
	printHits(hits3)

	fmt.Println("\n  Step 2 — Expand ±1 sentence and merge overlaps:")
	results3, err := Retrieve(ctx, store, q3, mockEmbed, 3, 1, 0)
	must(err)
	printWindows(results3)

	// -----------------------------------------------------------------------
	// MERGE DEMO — why interval-union matters
	// -----------------------------------------------------------------------
	header("WINDOW MERGE — why it matters")

	fmt.Println("  Query 1 produced these raw sentence hits:")
	for _, h := range hits1 {
		fmt.Printf("    %s (doc=%s pos=%d)\n", h.Sentence.ID, h.Sentence.DocID, h.Sentence.Pos)
	}
	fmt.Printf("\n  Without merge: %d separate ±1 windows (overlapping, repeating sentences)\n", len(hits1))
	fmt.Printf("  With    merge: %d contiguous block(s) — no sentence repeated\n", len(results1))
	fmt.Println("\n  → Adjacent hits collapse into one clean window. The LLM reads a")
	fmt.Println("    continuous passage centred on the match, not snapped to a fixed")
	fmt.Println("    section boundary (that's the difference from parent-child, 3.4.1).")

	fmt.Printf("\n%s\n", hr)
	fmt.Println("Done. Replace mockEmbed with a real provider (Voyage AI, Cohere, OpenAI, …)")
	fmt.Println("and replace the in-memory store with pgvector (sentences table with an")
	fmt.Println("embedding column + a plain (doc_id, pos) table for window expansion).")
	fmt.Printf("%s\n", hr)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
