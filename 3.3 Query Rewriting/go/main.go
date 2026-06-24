// main.go — runnable demo for all query-rewriting strategies.
//
//	go run .
//
// All LLM calls use a deterministic mock so no API key is required.
// Replace mockLLM with a real provider call in production.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// MOCK LLM
// ---------------------------------------------------------------------------
// In production replace this with a real call, e.g.:
//
//	import "github.com/anthropics/anthropic-sdk-go"
//	client := anthropic.NewClient()
//	func realLLM(ctx context.Context, prompt string) (string, error) {
//	    msg, _ := client.Messages.New(ctx, anthropic.MessageNewParams{
//	        Model:     anthropic.ModelClaude_Haiku_4_5,
//	        MaxTokens: 512,
//	        Messages:  []anthropic.MessageParam{{Role: "user", Content: prompt}},
//	    })
//	    return msg.Content[0].Text, nil
//	}

func mockLLM(_ context.Context, prompt string) (string, error) {
	p := strings.ToLower(prompt)

	switch {
	// HyDE — hypothetical answer passage.
	case strings.Contains(p, "factual passage") || strings.Contains(p, "directly answers"):
		return "Retrieval speed for large document collections can be significantly improved " +
			"by adding vector indexes (HNSW or IVFFlat in pgvector), tuning the similarity " +
			"metric, partitioning tables, and caching frequent query embeddings. " +
			"Approximate nearest-neighbour (ANN) search trades a small recall drop for " +
			"orders-of-magnitude speedup compared to exact search.", nil

	// Multi-Query — three paraphrase variants, one per line.
	case strings.Contains(p, "different phrasings") || strings.Contains(p, "paraphrase"):
		return "What techniques speed up document search in large databases?\n" +
			"How do I optimise vector search performance at scale?\n" +
			"Best practices for fast retrieval over millions of text chunks?", nil

	// Step-Back — broader conceptual question.
	case strings.Contains(p, "more general question") || strings.Contains(p, "step back"):
		return "What are the key architectural factors that determine database retrieval performance?", nil

	// Sub-query decomposition — atomic sub-questions.
	case strings.Contains(p, "sub-questions") || strings.Contains(p, "simpler"):
		return "What indexing strategies improve database retrieval speed?\n" +
			"How does approximate nearest-neighbour search compare to exact search?\n" +
			"What hardware configurations help with large-scale vector retrieval?\n" +
			"How can query embedding caching reduce retrieval latency?", nil

	// Contextual compression — standalone query.
	case strings.Contains(p, "standalone question") || strings.Contains(p, "conversation"):
		return "What are the scalability options available in PostgreSQL for handling large workloads?", nil

	default:
		// Fallback: return a minimal echo.
		return "What factors influence the performance of retrieval systems?", nil
	}
}

// ---------------------------------------------------------------------------
// MOCK RETRIEVER  (for RAG-Fusion demo)
// ---------------------------------------------------------------------------

func mockRetriever(_ context.Context, query string) ([]DocScore, error) {
	q := strings.ToLower(query)

	pools := map[string][]DocScore{
		"speed": {
			{"doc-ann-search", 0.95},
			{"doc-index-tuning", 0.85},
			{"doc-hardware", 0.80},
			{"doc-caching", 0.70},
		},
		"scale": {
			{"doc-partitioning", 0.90},
			{"doc-index-tuning", 0.86},
			{"doc-caching", 0.78},
			{"doc-ann-search", 0.72},
		},
		"optim": {
			{"doc-index-tuning", 0.91},
			{"doc-caching", 0.87},
			{"doc-ann-search", 0.82},
			{"doc-hardware", 0.68},
		},
		"default": {
			{"doc-index-tuning", 0.92},
			{"doc-ann-search", 0.88},
			{"doc-caching", 0.81},
			{"doc-hardware", 0.74},
			{"doc-partitioning", 0.65},
		},
	}

	for key, list := range pools {
		if key != "default" && strings.Contains(q, key) {
			return list, nil
		}
	}
	return pools["default"], nil
}

// ---------------------------------------------------------------------------
// DISPLAY HELPERS
// ---------------------------------------------------------------------------

const hr = "============================================================================"

func header(title string) { fmt.Printf("\n%s\n%s\n%s\n", hr, title, hr) }

func showQueries(qs []RewrittenQuery) {
	for i, q := range qs {
		meta, _ := json.Marshal(q.Metadata)
		fmt.Printf("  [%d] %q  %s\n", i, q.Query, meta)
	}
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

func main() {
	ctx := context.Background()

	query := "How can I improve the retrieval speed of my database for large document collections?"
	fmt.Printf("\nSample query:\n  %q\n", query)

	// 1. Keyword Expansion
	header("1. KEYWORD EXPANSION  (rule-based, no LLM)")
	kw := KeywordExpand(query, nil)
	fmt.Printf("  %q\n", kw.Query)

	// 2. HyDE
	header("2. HyDE — Hypothetical Document Embeddings")
	hyde, err := HyDE(ctx, query, mockLLM, 80)
	must(err)
	fmt.Printf("  Hypothetical passage to embed:\n  %q\n", hyde.Query)

	// 3. Multi-Query
	header("3. MULTI-QUERY  (numVariants=3)")
	mq, err := MultiQuery(ctx, query, mockLLM, 3)
	must(err)
	showQueries(mq)

	// 4. Step-Back
	header("4. STEP-BACK PROMPTING")
	sb, err := StepBack(ctx, query, mockLLM)
	must(err)
	showQueries(sb)

	// 5. Sub-query Decomposition
	header("5. SUB-QUERY DECOMPOSITION  (maxSubQueries=4)")
	sq, err := SubQueryDecompose(ctx, query, mockLLM, 4)
	must(err)
	showQueries(sq)

	// 6. RAG-Fusion
	header("6. RAG-FUSION  (numQueries=4, k=60)")
	fused, err := RAGFusion(ctx, query, mockLLM, mockRetriever, 4, 60)
	must(err)
	fmt.Println("  Fused & re-ranked documents:")
	for i, r := range fused {
		fmt.Printf("  [%d] docId=%q  rrfScore=%.5f  appearsIn=%d\n",
			i, r.DocID, r.RRFScore, r.AppearsIn)
	}

	// 7. Contextual Compression
	header("7. CONTEXTUAL COMPRESSION  (chat history → standalone query)")
	history := []ChatTurn{
		{Role: "user", Content: "What is PostgreSQL?"},
		{Role: "assistant", Content: "PostgreSQL is an open-source relational database known for ACID compliance and extensibility."},
		{Role: "user", Content: "What about its scalability options?"},
	}
	fmt.Println("  Chat history:")
	for _, t := range history {
		fmt.Printf("    %s: %s\n", t.Role, t.Content)
	}
	compressed, err := ContextualCompress(ctx, history, mockLLM)
	must(err)
	fmt.Printf("\n  Standalone query:\n  %q\n", compressed.Query)
	meta, _ := json.Marshal(compressed.Metadata)
	fmt.Printf("  %s\n", meta)

	fmt.Printf("\n%s\n", hr)
	fmt.Println("Done. Replace mockLLM / mockRetriever with real providers to use in production.")
	fmt.Printf("%s\n", hr)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
