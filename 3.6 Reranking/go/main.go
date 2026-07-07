// main.go — runnable demo for Reranking.
//
//	go run .
//
// No API key required. All models are lexical mocks:
//   - mockEmbed        : 6-dim concept vector (bi-encoder stand-in)
//   - mockCrossEncoder : joint query+doc term scoring (cross-encoder stand-in)
//   - mockTokenEmbed   : one concept vector per word (ColBERT stand-in)
//   - mockLlmJudge     : rubric grader 0–10 (LLM judge stand-in)
//
// In production, swap these for Cohere Rerank / Voyage rerank-2 / a BGE
// cross-encoder / a Claude judge. The reranker.go logic doesn't change.
//
// The demo walks the strategies in order:
//  1. Stage-1 bi-encoder baseline — fast but coarse; order is imperfect.
//  2. Cross-encoder rerank        — fixes the order.
//  3. Late interaction (MaxSim)   — the middle ground.
//  4. LLM rerank                  — pointwise vs pairwise vs listwise.
//  5. MMR                         — demotes near-duplicates.
//  6. RRF                         — fuses dense + keyword ranked lists.
//  7. Business rules              — recency + source authority.
package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// SAMPLE CORPUS
// ---------------------------------------------------------------------------
// Same knowledge-base theme as 3.5 (vector databases), built to expose the
// failure modes reranking fixes:
//   - "hnsw-tuning" is the true best answer for the demo query.
//   - "hnsw-rebuild" shares the query's keywords (HNSW, faster, recall) but
//     answers a DIFFERENT question — a classic bi-encoder trap.
//   - "hnsw-tuning-forum" near-duplicates the best answer (MMR fodder), and
//     is old + low-authority (business-rule fodder).

var corpus = []Chunk{
	{
		ID:       "hnsw-tuning",
		Text:     "To speed up HNSW queries without losing recall, tune ef_search: lower values are faster, higher values improve recall. Set M and ef_construction at build time for a better speed/recall frontier.",
		Metadata: map[string]string{"source": "docs", "ageDays": "30"},
	},
	{
		ID:       "hnsw-overview",
		Text:     "HNSW is a graph-based approximate nearest neighbour index. It builds a multi-layer proximity graph and offers high recall with real-time inserts.",
		Metadata: map[string]string{"source": "docs", "ageDays": "60"},
	},
	{
		ID:       "hnsw-tuning-forum",
		Text:     "Forum answer: basically you make HNSW faster by dropping ef_search, and you get recall back by raising it again. Also worth tweaking M and ef_construction when you build the index.",
		Metadata: map[string]string{"source": "forum", "ageDays": "700"},
	},
	{
		ID:       "ivfflat-tuning",
		Text:     "For IVFFlat, raising the probes parameter improves recall but slows queries; lowering it speeds them up. It is a different tradeoff curve than graph indexes.",
		Metadata: map[string]string{"source": "docs", "ageDays": "45"},
	},
	{
		ID:       "quantization",
		Text:     "Product quantization compresses vectors into compact codes, making search faster and indexes smaller, at the cost of a small recall loss unless you rerank with full-precision vectors.",
		Metadata: map[string]string{"source": "blog", "ageDays": "200"},
	},
	{
		ID:       "hnsw-rebuild",
		Text:     "Changelog: rebuilding an HNSW index is now 3x faster thanks to parallel graph construction. Recall is unaffected because the rebuild produces an identical graph.",
		Metadata: map[string]string{"source": "changelog", "ageDays": "10"},
	},
	{
		ID:       "normalization",
		Text:     "Embeddings should be L2-normalized before cosine similarity so that vector length does not distort semantic distance.",
		Metadata: map[string]string{"source": "docs", "ageDays": "90"},
	},
}

const query = "How can I make HNSW vector search faster without losing recall?"

// ---------------------------------------------------------------------------
// MOCK MODELS
// ---------------------------------------------------------------------------

// Concept dimensions for the bi-encoder mock. Coarse on purpose — a real
// bi-encoder is also coarse: it compresses a whole passage into one vector.
var concepts = [][]string{
	{"hnsw", "graph", "layer", "m,", "ef_search", "ef_construction"},
	{"faster", "fast", "speed", "latency", "slows", "quick"},
	{"recall", "accuracy", "quality"},
	{"index", "indexes", "ivfflat", "probes", "build", "rebuild"},
	{"vector", "vectors", "embedding", "embeddings", "quantization", "cosine"},
	{"changelog", "forum", "normalized", "parallel"},
}

var stopwords = map[string]bool{
	"how": true, "can": true, "i": true, "make": true, "the": true, "a": true,
	"an": true, "without": true, "to": true, "of": true, "is": true, "it": true,
	"and": true, "you": true, "by": true,
}

func tokenize(text string) []string {
	f := func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') && r != '_' && r != ','
	}
	return strings.FieldsFunc(strings.ToLower(text), f)
}

func contains(words []string, t string) bool {
	for _, w := range words {
		if w == t {
			return true
		}
	}
	return false
}

func mockEmbed(text string) []float64 {
	tokens := tokenize(text)
	vec := make([]float64, len(concepts))
	for i, words := range concepts {
		var hits float64
		for _, t := range tokens {
			if contains(words, t) {
				hits++
			}
		}
		vec[i] = hits / float64(len(tokens))
	}
	return vec
}

// Cross-encoder mock: because it sees query and doc TOGETHER, it can reward
// docs that contain the query's rare terms near each other, and penalize
// docs that merely mention the topic. Real cross-encoders learn this from
// data; the point here is the interface and the effect on ordering.
func mockCrossEncoder(q, doc string) float64 {
	var qTerms []string
	for _, t := range tokenize(q) {
		if !stopwords[t] {
			qTerms = append(qTerms, t)
		}
	}
	dTokens := tokenize(doc)
	dSet := make(map[string]bool, len(dTokens))
	for _, t := range dTokens {
		dSet[t] = true
	}

	// Term coverage: what fraction of query terms the doc actually contains.
	var hits float64
	for _, t := range qTerms {
		if dSet[t] {
			hits++
		}
	}
	score := hits / float64(len(qTerms))

	// Joint-attention stand-in: reward actionable answers ("tune X", "lower/
	// raise Y") over descriptions. A real cross-encoder picks this up from
	// relevance labels; we approximate with instruction verbs.
	actionable := []string{"tune", "set", "raising", "raise", "lowering", "lower", "dropping", "increase"}
	for _, t := range dTokens {
		if contains(actionable, t) {
			score += 0.35
			break
		}
	}

	// Penalize keyword-only matches: doc mentions the terms but is about a
	// different subject (our changelog trap talks about REBUILDING, not querying).
	if dSet["rebuild"] || dSet["rebuilding"] {
		score -= 0.4
	}
	return score
}

// ColBERT mock: one vector per token instead of one per document. Each token
// vector = concept dims (soft, shared across synonyms) + an identity dim from
// a small hash (so an EXACT token match scores higher than a same-concept
// match) — a crude imitation of contextualized token embeddings.
func mockTokenEmbed(text string) [][]float64 {
	const idDims = 8
	var out [][]float64
	for _, t := range tokenize(text) {
		if stopwords[t] {
			continue
		}
		vec := make([]float64, len(concepts)+idDims)
		for i, words := range concepts {
			if contains(words, t) {
				vec[i] = 0.6
			}
		}
		var hash int
		for _, r := range t {
			hash = (hash*31 + int(r)) % idDims
		}
		vec[len(concepts)+hash] = 1
		out = append(out, vec)
	}
	return out
}

// LLM judge mock: a deterministic "rubric" built on the cross-encoder score,
// snapped to a 0–10 integer — mimicking a temperature-0 grading prompt.
func mockLlmJudge(q, doc string) float64 {
	return math.Max(0, math.Min(10, math.Round(mockCrossEncoder(q, doc)*7)))
}

func mockLlmDuel(q, a, b string) bool {
	return mockCrossEncoder(q, a) >= mockCrossEncoder(q, b)
}

// Listwise mock: the model "reads" all passages in one prompt and emits an
// order. RankGPT would return "[1] > [3] > …"; we return the indices.
func mockLlmListRank(q string, docs []string) []int {
	idx := make([]int, len(docs))
	for i := range docs {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return mockCrossEncoder(q, docs[idx[a]]) > mockCrossEncoder(q, docs[idx[b]])
	})
	return idx
}

// Keyword retriever (BM25-lite) for the RRF demo — rank by raw query-term hits.
func keywordRetrieve(q string, chunks []Chunk, k int) []Scored {
	var qTerms []string
	for _, t := range tokenize(q) {
		if !stopwords[t] {
			qTerms = append(qTerms, t)
		}
	}
	var out []Scored
	for _, c := range chunks {
		dTokens := tokenize(c.Text)
		var score float64
		for _, t := range qTerms {
			for _, d := range dTokens {
				if d == t {
					score++
				}
			}
		}
		if score > 0 {
			out = append(out, Scored{Chunk: c, Score: score})
		}
	}
	sortByScoreDesc(out)
	return topN(out, k)
}

// ---------------------------------------------------------------------------
// DEMO
// ---------------------------------------------------------------------------

func printRanked(title string, ranked []Scored) {
	fmt.Printf("\n%s\n", title)
	for i, s := range ranked {
		fmt.Printf("  %d. [%.3f] %s (%s)\n", i+1, s.Score, s.Chunk.ID, s.Chunk.Metadata["source"])
	}
}

func main() {
	fmt.Printf("Query: %q\n", query)

	// 1. Stage 1 — bi-encoder over the whole corpus. Watch the shortlist: the
	// changelog trap makes the cut and off-topic quantization outranks docs
	// that actually answer the question, because one-vector-per-doc similarity
	// can't tell "mentions HNSW + faster + recall" apart from "tells you how".
	candidates := BiEncoderRetrieve(query, corpus, mockEmbed, 6)
	printRanked("1. STAGE 1 — bi-encoder retrieval (coarse, order imperfect)", candidates)

	// 2. Cross-encoder rerank — reads each (query, doc) pair jointly; the trap
	// drops, the actionable answer rises to #1.
	printRanked("2. CROSS-ENCODER rerank (production standard)",
		CrossEncoderRerank(query, candidates, mockCrossEncoder, 5))

	// 3. Late interaction — token-level MaxSim. Better than the bi-encoder
	// because each query token matches its best doc token independently.
	printRanked("3. LATE INTERACTION rerank (ColBERT-style MaxSim)",
		LateInteractionRerank(query, candidates, mockTokenEmbed, 5))

	// 4. LLM reranking, three flavors over the same candidates.
	printRanked("4a. LLM POINTWISE rerank (grade each doc 0–10)",
		LlmPointwiseRerank(query, candidates, mockLlmJudge, 5))
	printRanked("4b. LLM PAIRWISE rerank (round-robin duels, score = wins)",
		LlmPairwiseRerank(query, candidates, mockLlmDuel, 5))
	printRanked("4c. LLM LISTWISE rerank (RankGPT-style, one call ranks all)",
		LlmListwiseRerank(query, candidates, mockLlmListRank, 5))

	// 5. MMR — note the forum near-duplicate of the top answer gets pushed
	// down in favor of docs that add NEW information (quantization, ivfflat).
	printRanked("5. MMR rerank (λ=0.6 — relevance + diversity, duplicate demoted)",
		MmrRerank(query, candidates, mockEmbed, 0.6, 4))

	// 6. RRF — fuse the dense list with a keyword list using ranks only.
	denseList := BiEncoderRetrieve(query, corpus, mockEmbed, 6)
	keywordList := keywordRetrieve(query, corpus, 6)
	printRanked("6a. dense list (for fusion)", denseList)
	printRanked("6b. keyword list (for fusion)", keywordList)
	printRanked("6c. RRF fusion (k=60)", RrfRerank([][]Scored{denseList, keywordList}, 60, 5))

	// 7. Business rules — applied AFTER the cross-encoder. The 700-day-old
	// forum paraphrase sinks below fresher, authoritative docs even though its
	// pure relevance score was nearly identical.
	relevanceRanked := CrossEncoderRerank(query, candidates, mockCrossEncoder, 5)
	printRanked("7. BUSINESS-RULE rerank (recency half-life 365d, docs boosted, forum damped)",
		BusinessRuleRerank(relevanceRanked, BusinessRules{
			SourceWeights: map[string]float64{"docs": 1.2, "changelog": 1.0, "blog": 0.9, "forum": 0.6},
			HalfLifeDays:  365,
		}, 5))

	fmt.Println("\nProduction recipe: bi-encoder/hybrid top-50 → cross-encoder top-10 → " +
		"MMR/business rules → top-3..5 into the context window.")
}
