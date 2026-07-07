// reranker.go — every common reranking strategy, dependency-free.
//
// Reranking is STAGE 2 of retrieval. Stage 1 (bi-encoder / BM25 / hybrid)
// casts a wide, cheap net over the whole corpus and returns top-K candidates.
// Stage 2 re-scores ONLY those K candidates with a slower, more accurate
// model and reorders them before context construction.
//
//	corpus (1M docs) --stage 1: bi-encoder--> top 50 --stage 2: reranker--> top 5
//
// Why the split works:
//   - A bi-encoder embeds query and document SEPARATELY, then compares
//     vectors. Fast (documents pre-embedded offline), but the model never
//     sees query and document together, so it misses fine-grained relevance.
//   - A cross-encoder (or LLM judge) reads query + document TOGETHER in one
//     forward pass. Far more accurate, but O(K) model calls per query — you
//     can only afford it on a shortlist.
//
// Strategies implemented here:
//  1. Cross-encoder reranking    — the production standard.
//  2. Late interaction (ColBERT) — token-level MaxSim; the middle ground.
//  3. LLM pointwise reranking    — score each doc 0–10 independently.
//  4. LLM pairwise reranking     — "which of these two is better?" duels.
//  5. LLM listwise reranking     — RankGPT-style: rank the whole list at once.
//  6. MMR                        — diversity, not just relevance.
//  7. RRF                        — fuse ranked lists from several retrievers.
//  8. Business-rule reranking    — recency decay + source-authority boosts.
//
// All model calls are injected as function types (CrossEncoderFn, LlmJudgeFn,
// …) so this file stays dependency-free. main.go provides lexical mocks;
// production swaps in Cohere Rerank, Voyage rerank-2, a BGE cross-encoder,
// or a Claude/GPT judge without touching this logic.
package main

import (
	"math"
	"sort"
)

// ---------------------------------------------------------------------------
// TYPES
// ---------------------------------------------------------------------------

type Chunk struct {
	ID       string
	Text     string
	Metadata map[string]string
}

// Scored is a chunk with a score attached — the currency of every reranker.
type Scored struct {
	Chunk Chunk
	Score float64
}

// EmbedFn is bi-encoder style embedding: one vector per text.
type EmbedFn func(text string) []float64

// CrossEncoderFn sees query and document together and returns a relevance
// score. In production this is one forward pass of a model like
// BAAI/bge-reranker-v2-m3, or one API call to Cohere Rerank / Voyage rerank.
type CrossEncoderFn func(query, doc string) float64

// LlmJudgeFn is an LLM judge for pointwise scoring: returns relevance 0–10.
// In production: a small/cheap LLM with a rubric prompt, temperature 0.
type LlmJudgeFn func(query, doc string) float64

// LlmDuelFn is an LLM judge for pairwise comparison: returns true if A wins.
// In production: "Query: … Passage A: … Passage B: … Which passage answers
// the query better? Reply A or B."
type LlmDuelFn func(query, a, b string) bool

// LlmListRankFn is an LLM listwise ranker: given the query and all candidate
// texts, returns the indices in best-first order. This is the RankGPT
// pattern — one prompt with numbered passages, model replies "[3] > [1] > …".
type LlmListRankFn func(query string, docs []string) []int

// TokenEmbedFn is a token-level embedder for late interaction: one vector
// PER TOKEN.
type TokenEmbedFn func(text string) [][]float64

// ---------------------------------------------------------------------------
// SHARED MATH
// ---------------------------------------------------------------------------

func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func sortByScoreDesc(s []Scored) {
	sort.SliceStable(s, func(i, j int) bool { return s[i].Score > s[j].Score })
}

func topN(s []Scored, n int) []Scored {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// ---------------------------------------------------------------------------
// STAGE 1 — BI-ENCODER RETRIEVAL (the thing we rerank)
// ---------------------------------------------------------------------------
// Included so the demo is self-contained: this is the fast, coarse first
// pass. Note it compares one query vector against pre-computed doc vectors —
// no model ever reads query and document side by side.

func BiEncoderRetrieve(query string, corpus []Chunk, embed EmbedFn, k int) []Scored {
	qv := embed(query)
	scored := make([]Scored, 0, len(corpus))
	for _, c := range corpus {
		scored = append(scored, Scored{Chunk: c, Score: cosine(qv, embed(c.Text))})
	}
	sortByScoreDesc(scored)
	return topN(scored, k)
}

// ---------------------------------------------------------------------------
// 1. CROSS-ENCODER RERANKING — the production standard
// ---------------------------------------------------------------------------
// One scoring call per candidate: score = crossEncode(query, doc). The pair
// is processed jointly, so the model can see that "it" in the document
// refers to the thing the query asks about, that a negation flips relevance,
// that the doc mentions the query terms but answers a different question.
//
// Cost: K model calls (or one batched call). Never run it over the corpus.

func CrossEncoderRerank(query string, candidates []Scored, crossEncode CrossEncoderFn, n int) []Scored {
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		out = append(out, Scored{Chunk: c.Chunk, Score: crossEncode(query, c.Chunk.Text)})
	}
	sortByScoreDesc(out)
	return topN(out, n)
}

// ---------------------------------------------------------------------------
// 2. LATE INTERACTION (ColBERT-style MaxSim)
// ---------------------------------------------------------------------------
// The middle ground between bi- and cross-encoders. Documents are encoded
// offline like a bi-encoder — but into one vector PER TOKEN instead of one
// per document. At query time, each query token finds its best-matching
// document token (MaxSim), and those maxima are summed:
//
//	score(q, d) = Σ_{i ∈ q tokens} max_{j ∈ d tokens} sim(q_i, d_j)
//
// Fine-grained token-level matching (close to cross-encoder quality) with
// precomputable document representations (close to bi-encoder speed). The
// price is storage: ~100 vectors per document instead of 1.

func LateInteractionRerank(query string, candidates []Scored, tokenEmbed TokenEmbedFn, n int) []Scored {
	qTokens := tokenEmbed(query)
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		dTokens := tokenEmbed(c.Chunk.Text)
		var score float64
		for _, qv := range qTokens {
			var best float64
			for _, dv := range dTokens {
				best = math.Max(best, cosine(qv, dv))
			}
			score += best // MaxSim: each query token keeps only its best doc-token match
		}
		if len(qTokens) > 0 {
			score /= float64(len(qTokens))
		}
		out = append(out, Scored{Chunk: c.Chunk, Score: score})
	}
	sortByScoreDesc(out)
	return topN(out, n)
}

// ---------------------------------------------------------------------------
// 3. LLM POINTWISE RERANKING
// ---------------------------------------------------------------------------
// Ask an LLM to grade each (query, doc) pair independently on a rubric
// (0–10). Simple, parallelizable, and scores are absolute — usable as a
// confidence threshold ("if the best doc scores < 4, say I don't know").
// Weakness: LLMs are inconsistent graders; a 7 for one doc and a 6 for
// another may not reflect a real preference.

func LlmPointwiseRerank(query string, candidates []Scored, judge LlmJudgeFn, n int) []Scored {
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		out = append(out, Scored{Chunk: c.Chunk, Score: judge(query, c.Chunk.Text)})
	}
	sortByScoreDesc(out)
	return topN(out, n)
}

// ---------------------------------------------------------------------------
// 4. LLM PAIRWISE RERANKING
// ---------------------------------------------------------------------------
// Comparative judgments ("is A better than B?") are more reliable than
// absolute grades, at the cost of more calls. Full round-robin is O(K²)
// duels; each win earns a point and candidates are ranked by wins.
// Production systems cut the cost with a single bubble/insertion pass
// (O(K)) or a tournament bracket (O(K log K)).

func LlmPairwiseRerank(query string, candidates []Scored, duel LlmDuelFn, n int) []Scored {
	wins := make(map[string]float64, len(candidates))
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			a, b := candidates[i].Chunk, candidates[j].Chunk
			if duel(query, a.Text, b.Text) {
				wins[a.ID]++
			} else {
				wins[b.ID]++
			}
		}
	}
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		out = append(out, Scored{Chunk: c.Chunk, Score: wins[c.Chunk.ID]})
	}
	sortByScoreDesc(out)
	return topN(out, n)
}

// ---------------------------------------------------------------------------
// 5. LLM LISTWISE RERANKING (RankGPT pattern)
// ---------------------------------------------------------------------------
// Put ALL candidates in one prompt as numbered passages and ask the model to
// output a ranking ("[2] > [5] > [1] > …"). One call, and the model sees the
// candidates in context of each other — often the best quality per dollar.
// Weaknesses: position bias (models favor early passages — mitigate by
// shuffling), and the whole list must fit the context window (for large K,
// rank overlapping windows and merge — the sliding-window RankGPT trick).

func LlmListwiseRerank(query string, candidates []Scored, rankList LlmListRankFn, n int) []Scored {
	docs := make([]string, len(candidates))
	for i, c := range candidates {
		docs[i] = c.Chunk.Text
	}
	order := rankList(query, docs)
	out := make([]Scored, 0, len(candidates))
	for _, idx := range order {
		if idx >= 0 && idx < len(candidates) {
			// Score by rank position so downstream code still sees "higher = better".
			out = append(out, Scored{Chunk: candidates[idx].Chunk, Score: float64(len(candidates) - len(out))})
		}
	}
	return topN(out, n)
}

// ---------------------------------------------------------------------------
// 6. MMR — MAXIMAL MARGINAL RELEVANCE (diversity reranking)
// ---------------------------------------------------------------------------
// Every reranker above optimizes pure relevance, so the top-N is often five
// near-duplicates of the same paragraph. MMR greedily picks the next doc
// that is relevant to the query AND different from what's already picked:
//
//	MMR = λ · sim(doc, query) − (1 − λ) · max sim(doc, already-selected)
//
// λ = 1 → pure relevance; λ = 0 → pure diversity. 0.5–0.7 is a sane default.
// Typically run AFTER a relevance reranker, on its output.

func MmrRerank(query string, candidates []Scored, embed EmbedFn, lambda float64, n int) []Scored {
	qv := embed(query)
	type item struct {
		chunk Chunk
		vec   []float64
		rel   float64
	}
	pool := make([]item, 0, len(candidates))
	for _, c := range candidates {
		v := embed(c.Chunk.Text)
		pool = append(pool, item{chunk: c.Chunk, vec: v, rel: cosine(qv, v)})
	}

	var selected []item
	var out []Scored
	for len(out) < n && len(pool) > 0 {
		bestIdx, bestScore := 0, math.Inf(-1)
		for i, cand := range pool {
			var maxSim float64
			for _, s := range selected {
				maxSim = math.Max(maxSim, cosine(cand.vec, s.vec))
			}
			mmr := lambda*cand.rel - (1-lambda)*maxSim
			if mmr > bestScore {
				bestScore, bestIdx = mmr, i
			}
		}
		picked := pool[bestIdx]
		pool = append(pool[:bestIdx], pool[bestIdx+1:]...)
		selected = append(selected, picked)
		out = append(out, Scored{Chunk: picked.chunk, Score: bestScore})
	}
	return out
}

// ---------------------------------------------------------------------------
// 7. RRF — RECIPROCAL RANK FUSION (reranking across retrievers)
// ---------------------------------------------------------------------------
// When several retrievers each return a ranked list (dense, BM25, a second
// embedding model…), RRF merges them using only RANK positions — no score
// normalization needed, which is exactly why it's robust:
//
//	RRF(doc) = Σ_lists 1 / (k + rank_in_list)      k ≈ 60 by convention
//
// Also covered in 3.5 as hybrid-retrieval fusion; it appears here because
// fusing ranked lists IS a form of reranking, and it's the zero-model-call
// baseline every learned reranker must beat.

func RrfRerank(lists [][]Scored, k float64, n int) []Scored {
	scores := make(map[string]float64)
	chunks := make(map[string]Chunk)
	for _, list := range lists {
		for rank, s := range list {
			scores[s.Chunk.ID] += 1 / (k + float64(rank) + 1)
			chunks[s.Chunk.ID] = s.Chunk
		}
	}
	out := make([]Scored, 0, len(scores))
	for id, score := range scores {
		out = append(out, Scored{Chunk: chunks[id], Score: score})
	}
	sortByScoreDesc(out)
	return topN(out, n)
}

// ---------------------------------------------------------------------------
// 8. BUSINESS-RULE RERANKING (recency, authority, boosts)
// ---------------------------------------------------------------------------
// Pure relevance is not the whole product. A support bot should prefer the
// current docs page over a 2019 forum thread even if both match. This layer
// multiplies a relevance score by deterministic, explainable factors:
//
//	final = relevance × recencyDecay(ageDays) × sourceWeight(source)
//
// Runs LAST, on already-reranked results. Keep it multiplicative and gentle
// (weights near 1.0) so business rules tilt ties rather than overrule
// relevance entirely.

type BusinessRules struct {
	// e.g. {"docs": 1.2, "changelog": 1.0, "forum": 0.7} — unlisted sources get 1.0
	SourceWeights map[string]float64
	// score halves every HalfLifeDays of document age (Metadata["ageDays"])
	HalfLifeDays float64
}

func BusinessRuleRerank(candidates []Scored, rules BusinessRules, n int) []Scored {
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		ageDays := parseFloat(c.Chunk.Metadata["ageDays"])
		recency := math.Pow(0.5, ageDays/rules.HalfLifeDays)
		authority, ok := rules.SourceWeights[c.Chunk.Metadata["source"]]
		if !ok {
			authority = 1.0
		}
		out = append(out, Scored{Chunk: c.Chunk, Score: c.Score * recency * authority})
	}
	sortByScoreDesc(out)
	return topN(out, n)
}

func parseFloat(s string) float64 {
	var v float64
	for _, r := range s {
		if r < '0' || r > '9' {
			return v
		}
		v = v*10 + float64(r-'0')
	}
	return v
}
