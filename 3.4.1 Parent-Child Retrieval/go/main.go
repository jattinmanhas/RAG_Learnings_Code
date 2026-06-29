// main.go — runnable demo for Parent-Child Retrieval.
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
// SAMPLE DOCUMENT
// ---------------------------------------------------------------------------
// A realistic technical document split into three sections.  Each section
// becomes a PARENT chunk; each paragraph within a section becomes a CHILD chunk.
//
// Section length: ~150-200 words each (≈ 200-270 tokens).
// Child length:   ~50-70 words each  (≈ 65-90 tokens).
//
// In production these would be much larger (parents ~800 tokens, children ~150).

var parentTexts = []string{
	// Parent 0 — Vector Indexing Strategies
	`Vector Indexing Strategies

HNSW (Hierarchical Navigable Small World) is the most widely deployed approximate nearest-neighbour index for dense vector search. It builds a multi-layer proximity graph where each node is a vector. Queries traverse the graph from a random entry point at the top (sparse) layer and greedily descend to denser layers until the nearest neighbours are identified. HNSW achieves sub-linear query time — typically O(log n) — and can be updated incrementally without a rebuild.

IVFFlat (Inverted File with Flat quantization) partitions the vector space into Voronoi cells using k-means clustering. At index time each vector is assigned to its nearest centroid. At query time only the nprobe closest centroids are searched, reducing the comparison set from n to roughly n/nlist vectors. Unlike HNSW, IVFFlat requires a full rebuild when nlist changes and cannot be updated incrementally.

Choosing between HNSW and IVFFlat depends on your write pattern and memory budget. HNSW consumes more RAM (storing the graph edges) but supports real-time inserts and delivers higher recall at the same latency budget. IVFFlat uses less memory and is faster to build on a large static corpus, but recall drops sharply if nprobe is set too low. For most production RAG workloads with ongoing ingestion, HNSW is the safer default.`,

	// Parent 1 — Embedding Model Selection
	`Embedding Model Selection

Dense embedding models map text to a fixed-length float vector in a high-dimensional space (typically 768 to 3072 dimensions). Models such as text-embedding-3-large (OpenAI), embed-english-v3.0 (Cohere), and voyage-3 (Voyage AI) are trained on large corpora with contrastive objectives — they pull semantically similar passages close together and push dissimilar ones apart. The choice of model affects both retrieval quality and operational cost: larger models have higher dimensionality and slower throughput.

Sparse representations like BM25 or SPLADE score term frequency rather than semantic proximity. They excel at keyword-critical queries (product codes, proper nouns, rare terms) where dense models hallucinate similarity. Hybrid retrieval — running a dense retriever and a sparse retriever in parallel, then fusing their ranked lists with Reciprocal Rank Fusion — captures the strengths of both and consistently outperforms either alone on heterogeneous corpora.

Embedding dimensionality and normalization matter more than most practitioners expect. Truncating OpenAI's text-embedding-3-large from 3072 to 1536 dimensions loses less than 2 % NDCG@10 on MTEB while halving storage and index build time. Always L2-normalise embeddings before indexing when using cosine similarity — without normalisation, dot-product search and cosine search give different rankings, and most vector stores assume normalised inputs.`,

	// Parent 2 — Chunking Strategies
	`Chunking Strategies for RAG

Fixed-size chunking divides a document into non-overlapping windows of N tokens, ignoring sentence or paragraph boundaries. It is the simplest strategy to implement and produces uniform index sizes. The main risk is boundary cuts: a sentence split mid-clause loses meaning in both resulting chunks, degrading embedding quality. Fixed-size chunking is an acceptable baseline when documents are already structured (tables, code blocks, enumerated lists).

Semantic chunking groups sentences into chunks by measuring the embedding distance between consecutive sentences. When the distance spikes above a threshold, a new chunk begins. This keeps topically coherent passages together and avoids mid-topic splits, at the cost of variable chunk sizes. Variable sizes complicate batching for embedding APIs that charge per token, but the retrieval quality gain usually justifies it.

Overlap is a complementary technique: a sliding window moves forward by a stride smaller than the window size, so adjacent chunks share N tokens of context. Overlap ensures that information near a boundary appears in at least two chunks, reducing the chance of a critical sentence being isolated. A 20 % overlap (e.g. 20 shared tokens in a 100-token window) is a common starting point. Overlap increases index size proportionally to the overlap fraction, so it must be balanced against storage cost.`,
}

// ---------------------------------------------------------------------------
// MOCK EMBEDDING FUNCTION
// ---------------------------------------------------------------------------
// Maps any text to a 6-dimensional "concept vector" by counting keyword
// matches across six semantic dimensions.  Each dimension corresponds to a
// distinct topic cluster in the sample document above.
//
// Dimension legend:
//   [0] HNSW / graph-index concepts
//   [1] IVFFlat / centroid / Voronoi concepts
//   [2] embedding model / dimensionality concepts
//   [3] sparse / BM25 / hybrid retrieval concepts
//   [4] chunking / splitting / boundary concepts
//   [5] overlap / sliding window concepts
//
// The resulting vector is L2-normalised so cosine similarity is equivalent
// to the dot product.  In production, replace this with a real API call.
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
	{"hnsw", "graph", "hierarchical", "navigable", "layer", "node", "greedy", "descend", "entry"},
	{"ivfflat", "ivf", "voronoi", "centroid", "cluster", "nprobe", "nlist", "kmeans", "k-means"},
	{"embedding", "model", "dimension", "vector", "dense", "normalise", "normalize", "l2", "truncat"},
	{"sparse", "bm25", "splade", "hybrid", "keyword", "term", "frequency", "fusion", "reciprocal"},
	{"chunk", "chunking", "split", "fixed", "semantic", "boundary", "sentence", "window", "token"},
	{"overlap", "stride", "sliding", "adjacent", "shared", "20%", "20 %", "proportional"},
}

func mockEmbed(_ context.Context, text string) ([]float64, error) {
	lower := strings.ToLower(text)
	vec := make([]float64, len(conceptKeywords))
	for dim, keywords := range conceptKeywords {
		for _, kw := range keywords {
			// Count non-overlapping occurrences.
			count := strings.Count(lower, kw)
			vec[dim] += float64(count)
		}
	}
	// L2-normalise so cosine similarity equals the dot product.
	vec = l2Normalize(vec)
	return vec, nil
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

// truncate returns the first maxLen characters of s followed by "…" if longer.
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "…"
}

// printStore prints a summary of what was ingested.
func printStore(store *ParentChildStore) {
	fmt.Printf("\n  Ingested %d parent chunks, %d child chunks total.\n",
		len(store.parents), len(store.children))
	fmt.Println()

	for i, child := range store.children {
		fmt.Printf("  child[%d]  id=%-30s  parent=%s\n",
			i, child.ID, child.ParentID)
		fmt.Printf("            text=%q\n\n", truncate(child.Text, 80))
	}
}

// printResults prints the final parent chunks that would be fed to the LLM.
func printResults(results []ParentResult) {
	if len(results) == 0 {
		fmt.Println("  (no results)")
		return
	}
	for i, r := range results {
		fmt.Printf("  [%d] parentID  = %s\n", i, r.Parent.ID)
		fmt.Printf("       title     = %q\n", r.Parent.Title)
		fmt.Printf("       bestScore = %.4f\n", r.BestChildScore)
		fmt.Printf("       matched   = %v\n", r.MatchedChildren)
		fmt.Printf("       context   → %q\n\n", truncate(r.Parent.Text, 120))
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
	header("INGESTION PHASE — split document into parents and children")

	store := NewStore()
	// childSize = 50 words per child chunk (≈ 65 tokens) to keep the demo compact.
	// In production: parentSize ~800 tokens, childSize ~150 tokens.
	IngestDocument(store, "rag-guide", parentTexts, 50)
	printStore(store)

	// -----------------------------------------------------------------------
	// QUERY 1 — targets Parent 0 (Vector Indexing)
	// -----------------------------------------------------------------------
	header("QUERY 1 — Vector index trade-offs")

	q1 := "What are the differences between HNSW and IVFFlat for approximate nearest-neighbour search?"
	fmt.Printf("  Query: %q\n\n", q1)

	// Show which children are retrieved BEFORE parent mapping.
	childHits1, err := searchChildren(ctx, q1, store, mockEmbed, 4)
	must(err)
	fmt.Println("  Step 1 — Top-4 children from vector search:")
	for i, cs := range childHits1 {
		fmt.Printf("    [%d] score=%.4f  childID=%-32s  text=%q\n",
			i, cs.Score, cs.Child.ID, truncate(cs.Child.Text, 60))
	}

	// Full pipeline — map children → parents → deduplicate.
	fmt.Println()
	results1, err := Retrieve(ctx, store, q1, mockEmbed, 4, 0)
	must(err)
	fmt.Println("  Step 2 — Fetch parents, deduplicate:")
	printResults(results1)

	// -----------------------------------------------------------------------
	// QUERY 2 — targets Parent 1 (Embedding Models)
	// -----------------------------------------------------------------------
	header("QUERY 2 — Embedding model dimensionality")

	q2 := "How do embedding model dimensionality and L2 normalisation affect vector search quality?"
	fmt.Printf("  Query: %q\n\n", q2)

	childHits2, err := searchChildren(ctx, q2, store, mockEmbed, 4)
	must(err)
	fmt.Println("  Step 1 — Top-4 children from vector search:")
	for i, cs := range childHits2 {
		fmt.Printf("    [%d] score=%.4f  childID=%-32s  text=%q\n",
			i, cs.Score, cs.Child.ID, truncate(cs.Child.Text, 60))
	}

	fmt.Println()
	results2, err := Retrieve(ctx, store, q2, mockEmbed, 4, 0)
	must(err)
	fmt.Println("  Step 2 — Fetch parents, deduplicate:")
	printResults(results2)

	// -----------------------------------------------------------------------
	// QUERY 3 — cross-cutting query that spans two parents
	// -----------------------------------------------------------------------
	header("QUERY 3 — Cross-cutting query (spans two parent sections)")

	q3 := "What strategies reduce index size: chunking overlap, IVFFlat nlist, or dimensionality truncation?"
	fmt.Printf("  Query: %q\n\n", q3)

	childHits3, err := searchChildren(ctx, q3, store, mockEmbed, 6)
	must(err)
	fmt.Println("  Step 1 — Top-6 children from vector search:")
	for i, cs := range childHits3 {
		fmt.Printf("    [%d] score=%.4f  childID=%-32s  text=%q\n",
			i, cs.Score, cs.Child.ID, truncate(cs.Child.Text, 60))
	}

	fmt.Println()
	results3, err := Retrieve(ctx, store, q3, mockEmbed, 6, 0)
	must(err)
	fmt.Println("  Step 2 — Fetch parents, deduplicate:")
	fmt.Println("  (Multiple children from different parents → multiple full sections returned)")
	printResults(results3)

	// -----------------------------------------------------------------------
	// DEDUPLICATION DEMO
	// -----------------------------------------------------------------------
	header("DEDUPLICATION — why it matters")

	fmt.Println("  Query 1 retrieved these children:")
	for _, cs := range childHits1 {
		fmt.Printf("    childID=%-32s → parentID=%s\n", cs.Child.ID, cs.Child.ParentID)
	}
	fmt.Printf("\n  Without dedup: %d LLM context slots used\n", len(childHits1))
	fmt.Printf("  With    dedup: %d LLM context slots used (unique parents)\n", len(results1))
	fmt.Println("\n  → The LLM receives one full parent section per unique parent,")
	fmt.Println("    not a fragmented list of overlapping child excerpts.")

	fmt.Printf("\n%s\n", hr)
	fmt.Println("Done. Replace mockEmbed with a real provider (Voyage AI, Cohere, OpenAI, …)")
	fmt.Println("and replace the in-memory store with pgvector + a parent key-value table.")
	fmt.Printf("%s\n", hr)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
