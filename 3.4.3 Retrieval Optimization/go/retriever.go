// retriever.go — Retrieval Optimization for RAG
//
// WHAT THIS FILE COVERS
// ---------------------
// Sections 3.4.1 and 3.4.2 answered "what do I embed and how big is the
// context I return?" (parent-child, small-to-big). This file answers the next
// question every production RAG system hits: "given a corpus of chunks, how do
// I actually pull back the RIGHT ones — quickly, filtered, and tunably?"
//
// It demonstrates the four levers you reach for, in the order you reach for
// them:
//
//   1. TOP-K RETRIEVAL      — the baseline. Embed the query, rank every chunk
//                             by cosine similarity, keep the best K. Everything
//                             else is a refinement on top of this.
//
//   2. METADATA FILTERING   — narrow the candidate set BEFORE (or alongside)
//                             the vector search using structured fields:
//                             tenant, source, language, date, access level.
//                             This is about correctness and security, not just
//                             relevance — you must never return a chunk the
//                             user isn't allowed to see.
//
//   3. HYBRID RETRIEVAL     — dense vectors are great at meaning ("car" ~
//                             "automobile") but blind to exact tokens (product
//                             SKUs, error codes, rare names). Sparse keyword
//                             search (BM25) is the opposite. Hybrid retrieval
//                             runs BOTH and fuses the rankings so you get
//                             semantic recall AND exact-match precision.
//
//   4. RETRIEVAL TUNING     — none of the above has a single right setting.
//                             topK, the dense/sparse mix (alpha), the RRF
//                             constant, and a score floor are all knobs. This
//                             file makes every knob an explicit parameter and
//                             the demo (main.go) shows how turning each one
//                             changes what comes back.
//
// DESIGN NOTES
// ------------
//  • Dense scores come from an injectable EmbedFn (mock here, real provider in
//    production — see the comment on EmbedFn).
//  • Sparse scores come from a real, self-contained BM25 implementation so the
//    hybrid fusion is honest, not hand-waved.
//  • Fusion supports the two approaches you'll see in the wild:
//      - Reciprocal Rank Fusion (RRF): rank-based, scale-free, no tuning of
//        score normalisation. The safe default.
//      - Weighted score fusion (alpha): blends normalised dense & sparse
//        scores. More control, but you own the normalisation.
//  • Everything is in-memory for clarity. The inline SQL comments show the
//    pgvector equivalent you'd run in production.
package main

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// CORE TYPES
// ---------------------------------------------------------------------------

// Chunk is one retrievable unit — a passage plus the structured metadata you
// filter on. In production this is a row in a pgvector table.
type Chunk struct {
	ID       string            // stable identifier, e.g. "doc7-chunk-3"
	Text     string            // the passage that gets embedded and returned
	Metadata map[string]string // structured fields you can filter on
}

// Scored pairs a chunk with a score and, for transparency, the individual
// dense and sparse components that produced the final fused score. Surfacing
// the components is what makes retrieval TUNING possible — you can see whether
// a result won on meaning, on keywords, or on both.
type Scored struct {
	Chunk       Chunk
	Score       float64 // final score used for ranking (fused, in hybrid mode)
	DenseScore  float64 // cosine similarity component (0 if not computed)
	SparseScore float64 // BM25 component (0 if not computed)
	DenseRank   int      // 1-based rank in the dense list (0 = not present)
	SparseRank  int      // 1-based rank in the sparse list (0 = not present)
}

// EmbedFn converts text into a vector. Inject your real provider here.
//
// In production use (Anthropic pairs with Voyage AI for embeddings; Claude
// itself does not expose an embeddings endpoint):
//
//	import "github.com/anthropics/anthropic-sdk-go" // for generation
//	// Embeddings via Voyage AI (recommended alongside Claude), Cohere, or OpenAI:
//	resp, _ := voyageClient.Embed(ctx, &voyage.EmbedRequest{
//	    Input: []string{text},
//	    Model: "voyage-3",
//	})
//	return resp.Data[0].Embedding, nil
type EmbedFn func(ctx context.Context, text string) ([]float64, error)

// Filter is a metadata predicate: return true to KEEP the chunk as a candidate.
// Compose these to express "language == en AND access <= user's level".
type Filter func(Chunk) bool

// ---------------------------------------------------------------------------
// METADATA FILTERING (topic 2)
// ---------------------------------------------------------------------------
//
// Filtering is applied to candidates BEFORE scoring so the vector search never
// wastes work on — or worse, returns — chunks the user shouldn't see. In
// pgvector this is a WHERE clause on the same SELECT as the vector search:
//
//	SELECT id, text, 1 - (embedding <=> $q) AS score
//	FROM   chunks
//	WHERE  metadata->>'language' = 'en'
//	  AND  (metadata->>'access_level')::int <= $userLevel
//	ORDER  BY embedding <=> $q
//	LIMIT  $k;
//
// Doing it in one query lets Postgres use a partial/btree index on the metadata
// column, so you don't scan vectors you're about to discard.

// MatchAll keeps every chunk — the "no filter" default.
func MatchAll(Chunk) bool { return true }

// FieldEquals keeps chunks whose metadata[key] exactly equals value.
func FieldEquals(key, value string) Filter {
	return func(c Chunk) bool { return c.Metadata[key] == value }
}

// FieldIn keeps chunks whose metadata[key] is one of the allowed values.
func FieldIn(key string, values ...string) Filter {
	allowed := make(map[string]struct{}, len(values))
	for _, v := range values {
		allowed[v] = struct{}{}
	}
	return func(c Chunk) bool {
		_, ok := allowed[c.Metadata[key]]
		return ok
	}
}

// And composes filters with logical AND — a chunk must satisfy every filter.
// This is how you stack a relevance filter on top of a security filter.
func And(filters ...Filter) Filter {
	return func(c Chunk) bool {
		for _, f := range filters {
			if !f(c) {
				return false
			}
		}
		return true
	}
}

// applyFilter returns the subset of chunks that pass the predicate.
func applyFilter(chunks []Chunk, filter Filter) []Chunk {
	if filter == nil {
		filter = MatchAll
	}
	out := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		if filter(c) {
			out = append(out, c)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// DENSE RETRIEVAL (topic 1: Top-K)
// ---------------------------------------------------------------------------

// denseSearch embeds the query and every candidate, then ranks candidates by
// cosine similarity. Returns them sorted best-first (the caller truncates to K).
//
// In production the loop is a single pgvector query — the embeddings already
// live in the table, so you don't re-embed the corpus on every request. Here
// we embed on the fly because the mock embedder is free.
func denseSearch(
	ctx context.Context,
	query string,
	candidates []Chunk,
	embed EmbedFn,
) ([]Scored, error) {
	queryVec, err := embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	scored := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		vec, err := embed(ctx, c.Text)
		if err != nil {
			return nil, fmt.Errorf("embed chunk %s: %w", c.ID, err)
		}
		scored = append(scored, Scored{
			Chunk:      c,
			Score:      cosineSimilarity(queryVec, vec),
			DenseScore: cosineSimilarity(queryVec, vec),
		})
	}

	sort.SliceStable(scored, func(i, j int) bool {
		return scored[i].DenseScore > scored[j].DenseScore
	})
	for i := range scored {
		scored[i].DenseRank = i + 1
	}
	return scored, nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
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

// ---------------------------------------------------------------------------
// SPARSE RETRIEVAL — BM25 (the keyword half of hybrid)
// ---------------------------------------------------------------------------
//
// BM25 is the workhorse lexical ranking function behind Elasticsearch, Lucene,
// and Postgres full-text search. It scores a chunk for a query by summing, over
// each query term, that term's IDF (rare terms count more) times a saturating
// term-frequency factor (the 10th occurrence of a word adds little over the
// 2nd) with a length normalisation (long chunks don't win just by being long).
//
// We implement it directly so the hybrid demo is genuine. In production you'd
// let Postgres/Elasticsearch compute this; the fusion logic below is identical.

// bm25Params are the two classic BM25 knobs — themselves part of retrieval
// TUNING (topic 4).
type bm25Params struct {
	K1 float64 // term-frequency saturation (typical 1.2–2.0)
	B  float64 // length-normalisation strength (typical 0.75; 0 disables)
}

func defaultBM25() bm25Params { return bm25Params{K1: 1.5, B: 0.75} }

// bm25Index precomputes everything term-frequency-based so scoring a query is
// cheap. Built once at ingest, reused for every query.
type bm25Index struct {
	params  bm25Params
	docFreq map[string]int   // term -> number of chunks containing it
	tf      []map[string]int // per-chunk term -> count, aligned to chunks
	lens    []int            // per-chunk token count
	avgLen  float64          // mean chunk length in tokens
	n       int              // number of chunks
}

// tokenize lowercases and splits on non-alphanumerics — the minimal viable
// analyzer. Production systems add stemming, stop-word removal, and synonyms;
// the scoring math is unchanged.
func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
}

func buildBM25(chunks []Chunk, params bm25Params) *bm25Index {
	idx := &bm25Index{
		params:  params,
		docFreq: make(map[string]int),
		tf:      make([]map[string]int, len(chunks)),
		lens:    make([]int, len(chunks)),
		n:       len(chunks),
	}
	var totalLen int
	for i, c := range chunks {
		counts := make(map[string]int)
		toks := tokenize(c.Text)
		for _, t := range toks {
			counts[t]++
		}
		idx.tf[i] = counts
		idx.lens[i] = len(toks)
		totalLen += len(toks)
		for term := range counts { // document frequency = presence, not count
			idx.docFreq[term]++
		}
	}
	if idx.n > 0 {
		idx.avgLen = float64(totalLen) / float64(idx.n)
	}
	return idx
}

// idf uses the BM25 "probabilistic" IDF with a +1 to keep it non-negative.
func (idx *bm25Index) idf(term string) float64 {
	df := idx.docFreq[term]
	if df == 0 {
		return 0
	}
	return math.Log(1 + (float64(idx.n)-float64(df)+0.5)/(float64(df)+0.5))
}

// scoreDoc computes the BM25 score of chunk i for the given query terms.
func (idx *bm25Index) scoreDoc(i int, queryTerms []string) float64 {
	if i < 0 || i >= idx.n {
		return 0
	}
	k1, b := idx.params.K1, idx.params.B
	var score float64
	for _, term := range queryTerms {
		f := float64(idx.tf[i][term])
		if f == 0 {
			continue
		}
		norm := 1 - b + b*float64(idx.lens[i])/idx.avgLen
		score += idx.idf(term) * (f * (k1 + 1)) / (f + k1*norm)
	}
	return score
}

// sparseSearch ranks the given candidates by BM25 against the query. It takes
// the candidate subset (already metadata-filtered) plus the index positions so
// it can reuse the precomputed term frequencies.
func sparseSearch(query string, candidates []Chunk, idx *bm25Index, pos map[string]int) []Scored {
	terms := tokenize(query)
	scored := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		i, ok := pos[c.ID]
		if !ok {
			continue
		}
		scored = append(scored, Scored{
			Chunk:       c,
			Score:       idx.scoreDoc(i, terms),
			SparseScore: idx.scoreDoc(i, terms),
		})
	}
	sort.SliceStable(scored, func(i, j int) bool {
		return scored[i].SparseScore > scored[j].SparseScore
	})
	for i := range scored {
		scored[i].SparseRank = i + 1
	}
	return scored
}

// ---------------------------------------------------------------------------
// FUSION — combining dense + sparse (topic 3: Hybrid)
// ---------------------------------------------------------------------------

// FusionMode selects how the two ranked lists are combined.
type FusionMode int

const (
	// FusionRRF — Reciprocal Rank Fusion. Combines by RANK, not score, so it
	// needs no score normalisation and is robust when dense and sparse scores
	// live on wildly different scales. The production-safe default.
	FusionRRF FusionMode = iota
	// FusionWeighted — blend min-max-normalised dense & sparse SCORES with a
	// weight alpha (alpha=1 → pure dense, alpha=0 → pure sparse). More control,
	// but you own the normalisation and it's sensitive to score outliers.
	FusionWeighted
)

// fuseRRF implements Reciprocal Rank Fusion:
//
//	score(chunk) = Σ  1 / (rrfK + rank_in_list)
//
// summed over the dense and sparse lists the chunk appears in. rrfK (typically
// 60) damps the contribution of top ranks so no single list dominates. A chunk
// that ranks well in BOTH lists beats one that ranks great in only one — which
// is exactly the hybrid behaviour we want.
func fuseRRF(dense, sparse []Scored, rrfK float64) []Scored {
	if rrfK <= 0 {
		rrfK = 60
	}
	agg := make(map[string]*Scored)

	accumulate := func(list []Scored, isDense bool) {
		for rank, s := range list {
			cur, ok := agg[s.Chunk.ID]
			if !ok {
				c := s.Chunk
				cur = &Scored{Chunk: c}
				agg[s.Chunk.ID] = cur
			}
			cur.Score += 1.0 / (rrfK + float64(rank+1))
			if isDense {
				cur.DenseScore = s.DenseScore
				cur.DenseRank = rank + 1
			} else {
				cur.SparseScore = s.SparseScore
				cur.SparseRank = rank + 1
			}
		}
	}
	accumulate(dense, true)
	accumulate(sparse, false)

	return sortAgg(agg)
}

// fuseWeighted min-max normalises each list to [0,1] and blends:
//
//	score = alpha * denseNorm + (1 - alpha) * sparseNorm
func fuseWeighted(dense, sparse []Scored, alpha float64) []Scored {
	alpha = clamp01(alpha)
	dNorm := minMaxNormalize(dense, func(s Scored) float64 { return s.DenseScore })
	sNorm := minMaxNormalize(sparse, func(s Scored) float64 { return s.SparseScore })

	agg := make(map[string]*Scored)
	ensure := func(s Scored) *Scored {
		cur, ok := agg[s.Chunk.ID]
		if !ok {
			c := s.Chunk
			cur = &Scored{Chunk: c}
			agg[s.Chunk.ID] = cur
		}
		return cur
	}
	for i, s := range dense {
		cur := ensure(s)
		cur.DenseScore = s.DenseScore
		cur.DenseRank = i + 1
		cur.Score += alpha * dNorm[s.Chunk.ID]
	}
	for i, s := range sparse {
		cur := ensure(s)
		cur.SparseScore = s.SparseScore
		cur.SparseRank = i + 1
		cur.Score += (1 - alpha) * sNorm[s.Chunk.ID]
	}
	return sortAgg(agg)
}

func minMaxNormalize(list []Scored, get func(Scored) float64) map[string]float64 {
	out := make(map[string]float64, len(list))
	if len(list) == 0 {
		return out
	}
	min, max := math.Inf(1), math.Inf(-1)
	for _, s := range list {
		v := get(s)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	span := max - min
	for _, s := range list {
		if span == 0 {
			out[s.Chunk.ID] = 0 // all equal → no discriminating signal
		} else {
			out[s.Chunk.ID] = (get(s) - min) / span
		}
	}
	return out
}

func sortAgg(agg map[string]*Scored) []Scored {
	out := make([]Scored, 0, len(agg))
	for _, s := range agg {
		out = append(out, *s)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Chunk.ID < out[j].Chunk.ID // stable tie-break
	})
	return out
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// ---------------------------------------------------------------------------
// THE STORE + PUBLIC RETRIEVAL API (topic 4: everything is a tunable knob)
// ---------------------------------------------------------------------------

// Store holds the corpus, the precomputed BM25 index, and a lookup from chunk
// ID to its index position. Build it once, query it many times.
type Store struct {
	chunks []Chunk
	bm25   *bm25Index
	pos    map[string]int // chunk ID -> position in chunks / bm25 arrays
}

// NewStore ingests the corpus and builds the BM25 index up front.
func NewStore(chunks []Chunk, bm25Cfg bm25Params) *Store {
	pos := make(map[string]int, len(chunks))
	for i, c := range chunks {
		pos[c.ID] = i
	}
	return &Store{
		chunks: chunks,
		bm25:   buildBM25(chunks, bm25Cfg),
		pos:    pos,
	}
}

// Options bundles every retrieval knob in one place. This IS "retrieval
// tuning": each field is a lever, and main.go sweeps them to show the effect.
type Options struct {
	TopK      int        // how many results to return (topic 1)
	Filter    Filter     // metadata predicate; nil = MatchAll (topic 2)
	Hybrid    bool       // false = dense only; true = dense + BM25 (topic 3)
	Fusion    FusionMode // RRF or weighted, when Hybrid (topic 3)
	Alpha     float64    // dense weight for FusionWeighted (topic 4)
	RRFK      float64    // RRF damping constant, default 60 (topic 4)
	Threshold float64    // drop results whose final Score < Threshold (topic 4)
	// CandidateK caps how many per-retriever candidates feed fusion. Larger =
	// better recall, more compute — the classic retrieve-wide-then-narrow knob.
	CandidateK int
}

// DefaultOptions is a sensible production starting point: hybrid + RRF, top 5.
func DefaultOptions() Options {
	return Options{
		TopK:       5,
		Filter:     MatchAll,
		Hybrid:     true,
		Fusion:     FusionRRF,
		Alpha:      0.5,
		RRFK:       60,
		Threshold:  0,
		CandidateK: 20,
	}
}

// Retrieve runs the full optimised pipeline:
//
//	metadata filter → dense (+ sparse) search → fuse → threshold → top-K
//
// Every stage is governed by Options, so the same function covers plain top-K,
// filtered search, and full hybrid retrieval depending on how you configure it.
func (s *Store) Retrieve(ctx context.Context, query string, embed EmbedFn, opts Options) ([]Scored, error) {
	if opts.TopK <= 0 {
		opts.TopK = 5
	}
	if opts.CandidateK <= 0 {
		opts.CandidateK = 20
	}
	if opts.Filter == nil {
		opts.Filter = MatchAll
	}

	// Stage 1 — metadata filter (topic 2): shrink the candidate set first.
	candidates := applyFilter(s.chunks, opts.Filter)
	if len(candidates) == 0 {
		return nil, nil
	}

	// Stage 2 — dense retrieval (topic 1), truncated to CandidateK.
	dense, err := denseSearch(ctx, query, candidates, embed)
	if err != nil {
		return nil, err
	}
	dense = truncate(dense, opts.CandidateK)

	// Dense-only path: no fusion, just threshold + top-K.
	if !opts.Hybrid {
		return finalize(dense, opts), nil
	}

	// Stage 3 — sparse retrieval (topic 3), same candidate set + CandidateK.
	sparse := sparseSearch(query, candidates, s.bm25, s.pos)
	sparse = truncate(sparse, opts.CandidateK)

	// Stage 4 — fuse the two rankings (topic 3).
	var fused []Scored
	switch opts.Fusion {
	case FusionWeighted:
		fused = fuseWeighted(dense, sparse, opts.Alpha)
	default:
		fused = fuseRRF(dense, sparse, opts.RRFK)
	}

	// Stage 5 — threshold + top-K (topic 4).
	return finalize(fused, opts), nil
}

// finalize applies the score threshold then truncates to TopK.
func finalize(list []Scored, opts Options) []Scored {
	out := make([]Scored, 0, len(list))
	for _, s := range list {
		if s.Score >= opts.Threshold {
			out = append(out, s)
		}
	}
	return truncate(out, opts.TopK)
}

func truncate(list []Scored, k int) []Scored {
	if k > 0 && len(list) > k {
		return list[:k]
	}
	return list
}
