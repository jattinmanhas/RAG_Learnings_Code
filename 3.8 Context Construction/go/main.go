// main.go — runnable demo for Context Construction.
//
//	go run .
//
// No API key required. Mocks:
//   - countTokens : chars/4 (stand-in for tiktoken / count-tokens API)
//   - summarize   : first sentence + query terms (stand-in for a Haiku call)
//
// A retrieval result for one query is carried through every stage: packing
// under a tight budget, the four orderings, the compression cascade,
// citation-ready formatting, and finally the complete pipeline end to end.
package main

import (
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// SAMPLE RETRIEVAL RESULT — same vector-database theme as 3.5–3.7, but now
// every chunk carries the provenance a citation needs (source + loc + ordinal)
// and the list contains the messes real retrieval produces: a near-duplicate,
// one very long chunk, and several hits from the same file.
// ---------------------------------------------------------------------------

var hits = []Chunk{
	{
		ID: "c1", Score: 0.91, Source: "docs/hnsw-tuning.md", Loc: "L12-L28", Ordinal: 2,
		Text: "To make HNSW search faster without hurting recall, raise efSearch gradually and re-measure recall on a held-out query set. efSearch is a pure latency/recall dial at query time; M and efConstruction are build-time and require a reindex.",
	},
	{
		ID: "c2", Score: 0.88, Source: "docs/hnsw-tuning.md", Loc: "L30-L44", Ordinal: 3,
		Text: "A practical tuning loop: fix M, sweep efSearch over 40, 80, 160, 320 and plot recall@10 against p95 latency. Stop at the knee of the curve. Raising M helps recall on high-dimensional data but costs memory and build time.",
	},
	{
		// NEAR-DUPLICATE of c1 — overlapping-window chunking produced both.
		ID: "c3", Score: 0.86, Source: "docs/hnsw-tuning.md", Loc: "L12-L27", Ordinal: 2,
		Text: "To make HNSW search faster without hurting recall, raise efSearch gradually and re-measure recall on a held-out query set. efSearch is a pure latency/recall dial at query time; M and efConstruction are build-time settings.",
	},
	{
		ID: "c4", Score: 0.72, Source: "docs/hnsw-basics.md", Loc: "L1-L9", Ordinal: 1,
		Text: "HNSW is a graph-based index for approximate nearest neighbour search, built from a hierarchy of navigable small-world graphs.",
	},
	{
		// The long one — high value per chunk, terrible value per token.
		ID: "c5", Score: 0.69, Source: "docs/pgvector.md", Loc: "L60-L120", Ordinal: 5,
		Text: "pgvector supports cosine distance, L2 distance and inner product operators, each with its own operator class. Index selection follows the operator you query with: vector_cosine_ops for cosine, vector_l2_ops for Euclidean, vector_ip_ops for inner product. Building an HNSW index in pgvector takes maintenance_work_mem into account, so raise it before a large build or the build will spill to disk and take hours. The ef_search parameter is set per session with SET hnsw.ef_search, which means an application pool must set it on every connection or fall back to the default of 40. IVFFlat remains available and builds far faster, but its recall degrades as the table grows unless you reindex after significant inserts. For production workloads on tables above a million rows, HNSW is the default recommendation despite the longer build.",
	},
	{
		ID: "c6", Score: 0.64, Source: "docs/hnsw-tuning.md", Loc: "L46-L58", Ordinal: 4,
		Text: "Recall is measured against exact brute-force search on the same query set. Never tune efSearch against production traffic without a ground-truth set; you will optimise latency and silently lose answers.",
	},
	{
		ID: "c7", Score: 0.58, Source: "docs/ivf.md", Loc: "L1-L14", Ordinal: 1,
		Text: "IVF indexes partition vectors into clusters and probe a subset at query time. Probing more lists improves recall at higher latency. Recall is measured against exact brute-force search on the same query set.",
	},
	{
		ID: "c8", Score: 0.41, Source: "docs/embeddings.md", Loc: "L20-L31", Ordinal: 2,
		Text: "Choosing an embedding model means balancing dimensionality, cost and retrieval quality. Switching models requires re-embedding the whole corpus and re-calibrating every threshold you tuned.",
	},
}

const query = "How do I make HNSW search faster without hurting recall?"

// ---------------------------------------------------------------------------
// MOCKS — swap for a real tokenizer and a real LLM
// ---------------------------------------------------------------------------

// ~4 characters per token is the standard English rule of thumb. Real systems
// must use the actual tokenizer: guessing low truncates, guessing high wastes
// budget on every single request.
func countTokens(text string) int { return (len([]rune(text)) + 3) / 4 }

// summarize stands in for "summarise this passage with respect to the query".
func summarize(text, q string) string {
	qs := wordSet(q)
	first := text
	if s := splitSentences(text); len(s) > 0 {
		first = s[0]
	}
	keys := []string{}
	seen := map[string]bool{}
	for _, w := range nonWord.Split(strings.ToLower(text), -1) {
		if len(w) > 4 && qs[w] && !seen[w] && len(keys) < 4 {
			seen[w] = true
			keys = append(keys, w)
		}
	}
	if len(keys) == 0 {
		keys = []string{"n/a"}
	}
	r := []rune(first)
	if len(r) > 90 {
		r = r[:90]
	}
	return fmt.Sprintf("%s… (key: %s)", strings.TrimSpace(string(r)), strings.Join(keys, ", "))
}

func section(title string) {
	line := strings.Repeat("=", 78)
	fmt.Printf("\n%s\n%s\n%s\n", line, title, line)
}

func ids(cs []Chunk) string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, c.ID)
	}
	return strings.Join(out, ", ")
}

func idsWithScores(cs []Chunk) string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, fmt.Sprintf("%s(%.2f)", c.ID, c.Score))
	}
	return strings.Join(out, " → ")
}

func show(label string, cs []Chunk) {
	fmt.Printf("%s: %3d tok  [%s]\n", label, TotalTokens(cs, countTokens), ids(cs))
}

func main() {
	fmt.Printf("query: %q\n", query)
	fmt.Printf("retrieved: %d chunks, %d tokens raw\n", len(hits), TotalTokens(hits, countTokens))

	// --- 1. Context Packing --------------------------------------------------
	section("1. CONTEXT PACKING — fitting maximum relevant content in the budget")

	budget := EvidenceBudget(BudgetSpec{
		ContextWindow: 1000, // deliberately tiny so the trade-offs are visible
		SystemTokens:  120,
		QueryTokens:   countTokens(query),
		AnswerReserve: 400,
	})
	fmt.Printf("1a evidence budget      : %d tok (window 1000 − system 120 − query − answer 400 − margin)\n\n", budget)

	show("1b greedy (by score)   ", PackGreedy(hits, budget, countTokens))
	show("1c density (score/tok) ", PackByDensity(hits, budget, countTokens, 1))
	show("1d after dedupe        ", Dedupe(hits, 0.8))
	show("1e cap 2 per source    ", CapPerSource(hits, 2))
	show("1f tiered (2 full, rest summarised)",
		PackTiered(hits, budget, countTokens, 2, func(c Chunk) string { return summarize(c.Text, query) }))
	fmt.Println("\n   note: greedy skips c5 (the 209-tok pgvector chunk) and fits c6 + c7 in its place —")
	fmt.Println("   that single 'continue' instead of 'break' is worth two extra chunks of evidence.")
	fmt.Println("   dedupe (1d) is free budget: c3 is a near-copy of c1 and buys ~56 tok back.")

	// --- 2. Context Ordering -------------------------------------------------
	section("2. CONTEXT ORDERING — highest relevance at top and bottom (not middle)")

	packed := PackGreedy(Dedupe(hits, 0.8), budget, countTokens)
	fmt.Printf("2a relevance desc  : %s\n", idsWithScores(OrderByRelevance(packed)))
	fmt.Println("                     ↑ strongest evidence in the head, WEAKEST in the strong tail slot")
	fmt.Printf("2b bookend (V)     : %s\n", idsWithScores(OrderBookend(packed)))
	fmt.Println("                     ↑ ranks 1 & 2 at both attention peaks, weakest buried mid-context")
	fmt.Printf("2c document order  : %s\n", idsWithScores(OrderByDocument(packed)))
	fmt.Println("                     ↑ contiguous runs per file, files ranked by their best chunk")
	fmt.Printf("2d grouped bookend : %s\n", idsWithScores(OrderGroupedBookend(packed)))
	fmt.Println("                     ↑ coherent per-file runs, whole groups placed at the peaks")

	layout := QueryAdjacentLayout(query, "<context>…</context>", "…")
	fmt.Println("\n2e query-adjacent  : instructions+query → context → query restated LAST")
	fmt.Printf("                     final line: %q\n", strings.Split(layout.Post, "\n")[0])

	// --- 3. Context Compression ----------------------------------------------
	section("3. CONTEXT COMPRESSION — summarising or trimming before injection")

	var long Chunk
	for _, c := range hits {
		if c.ID == "c5" {
			long = c
		}
	}
	clip := func(s string, n int) string {
		r := []rune(s)
		if len(r) > n {
			return string(r[:n]) + "…"
		}
		return s
	}
	fmt.Printf("original c5            : %d tok\n", countTokens(long.Text))
	ext := ExtractiveCompress(long.Text, query, 2)
	fmt.Printf("3a extractive (2 sent) : %d tok\n", countTokens(ext))
	fmt.Printf("   → %q\n", clip(ext, 110))
	fmt.Printf("3b truncate @40 tok    : %q\n", clip(TruncateToTokens(long.Text, 40, countTokens), 110))
	pruned := PruneRedundantSentences([]Chunk{hits[5], hits[6]})
	fmt.Printf("3c cross-chunk pruning : c6+c7 %d → %d tok\n",
		countTokens(hits[5].Text)+countTokens(hits[6].Text), TotalTokens(pruned, countTokens))
	fmt.Println("   → the shared sentence (\"Recall is measured against exact brute-force…\") appears once")
	fmt.Printf("3d abstractive (LLM)   : %q\n", summarize(long.Text, query))
	fmt.Println("   → paraphrase: cheap, lossy, NOT quotable — never do this to your top hit")

	cascade := CompressionCascade(hits, query, budget, countTokens, summarize, 1)
	fmt.Println("\n3e cascade (stops as soon as it fits):")
	for _, s := range cascade.Stages {
		fmt.Printf("   %s\n", s)
	}

	// --- 4. Citation-ready formatting ----------------------------------------
	section("4. CITATION-READY FORMATTING (the handoff to 3.9 Citations)")

	top3 := OrderBookend(packed)
	if len(top3) > 3 {
		top3 = top3[:3]
	}
	formatted := FormatForCitation(top3, countTokens, FormatOptions{Style: "xml"})
	fmt.Println(formatted.Text)
	fmt.Printf("\ninstructions:\n%s\n", CitationInstructions())

	modelAnswer := "Raise efSearch gradually and re-measure recall on a held-out set [1]. " +
		"Sweep 40/80/160/320 and stop at the knee [2]. Also enable turbo mode [7]."
	used, usedNums, invalid := ResolveCitations(modelAnswer, formatted.Citations)
	fmt.Println("\nresolving the answer's markers:")
	for i, c := range used {
		fmt.Printf("   ✓ [%d] → %s:%s\n", usedNums[i], c.Source, c.Loc)
	}
	for _, n := range invalid {
		fmt.Printf("   ✗ [%d] → no such source (hallucinated citation — 3.9 rejects the sentence)\n", n)
	}

	// --- 5. The Complete Context Construction Pipeline ------------------------
	section("5. THE COMPLETE CONTEXT CONSTRUCTION PIPELINE")

	result := BuildContext(query, hits, BuildConfig{
		ContextWindow:   1000,
		SystemTokens:    120,
		AnswerReserve:   400,
		MaxPerSource:    3,
		DedupeThreshold: 0.8,
		ProtectTop:      1,
		Ordering:        "grouped-bookend",
		Style:           "xml",
	}, countTokens, summarize)

	for _, line := range result.Trace {
		fmt.Println(line)
	}
	fmt.Printf("\n%s\nFINAL PROMPT\n%s\n", strings.Repeat("-", 78), strings.Repeat("-", 78))
	fmt.Println(result.Prompt)
}
