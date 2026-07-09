// confidence.go — Retrieval confidence signals for RAG.
//
// Three families, cheapest → most expensive:
//  1. Absolute thresholding   (crude, corpus-specific)
//  2. Relative/distributional (more robust, still free)
//  3. LLM-based groundedness  (most robust, costs LLM calls)
//
// plus the Complete Confidence Pipeline that composes them cheapest-first.
//
// Everything is model-agnostic: scores come in as plain numbers, LLM judges
// come in as injected functions, so the mocks in main.go can be swapped for
// real embedding / Claude calls without touching this file.
package main

import (
	"fmt"
	"math"
	"sort"
)

// Hit is a retrieved chunk with its similarity score (higher = more similar).
type Hit struct {
	ID    string
	Text  string
	Score float64
}

type Verdict string

const (
	VerdictAnswer Verdict = "ANSWER"
	VerdictHedge  Verdict = "HEDGE"
	VerdictRefuse Verdict = "REFUSE"
)

// ===========================================================================
// 1. ABSOLUTE THRESHOLDING (crude, corpus-specific)
// ===========================================================================
// Compare raw scores to fixed cutoffs. Free, trivially simple — but the right
// cutoff depends on the embedding model, the distance metric, and the corpus.
// A cosine of 0.30 is "unrelated" for one model and "quite similar" for
// another. Never ship these without calibrating on YOUR data.

// SimilarityFloor (1a) drops every hit below tau.
func SimilarityFloor(hits []Hit, tau float64) []Hit {
	out := make([]Hit, 0, len(hits))
	for _, h := range hits {
		if h.Score >= tau {
			out = append(out, h)
		}
	}
	return out
}

// Top1Floor (1b): if even the best hit is weak, refuse outright.
// The cheapest possible "I don't know" gate.
func Top1Floor(hits []Hit, tau float64) bool {
	return len(hits) > 0 && hits[0].Score >= tau
}

// CountAboveThreshold (1c) requires at least minCount decent hits, so a
// single lucky keyword match can't carry an answer alone.
func CountAboveThreshold(hits []Hit, tau float64, minCount int) bool {
	n := 0
	for _, h := range hits {
		if h.Score >= tau {
			n++
		}
	}
	return n >= minCount
}

// CalibrateThreshold (1d) derives tau from data instead of guessing.
// Given similarity scores of known-IRRELEVANT query/chunk pairs (a small
// labelled dev set), pick the given percentile (e.g. 0.95) as tau: anything
// scoring above it is unlikely to be background noise. This is the only
// defensible way to set an absolute cutoff — and it must be redone whenever
// the embedding model or corpus changes.
func CalibrateThreshold(irrelevantPairScores []float64, percentile float64) float64 {
	sorted := append([]float64(nil), irrelevantPairScores...)
	sort.Float64s(sorted)
	idx := int(percentile * float64(len(sorted)))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// RerankerFloor (1e): same rule, better input. Cross-encoder / pointwise-LLM
// scores (see 3.6) are far better calibrated than raw cosine, so an absolute
// floor on THEM transfers much better.
func RerankerFloor(hits []Hit, rerankScore func(Hit) float64, tau float64) []Hit {
	out := make([]Hit, 0, len(hits))
	for _, h := range hits {
		if rerankScore(h) >= tau {
			out = append(out, h)
		}
	}
	return out
}

// ===========================================================================
// 2. RELATIVE / DISTRIBUTIONAL SIGNALS (more robust)
// ===========================================================================
// Ignore the absolute values; look at the SHAPE of the top-K score
// distribution. Shape transfers across models and corpora much better:
// "one hit towers over the rest" means the same thing everywhere.

// Top1Top2Gap (2a): a clear winner separates from the pack. Big gap ⇒ one
// chunk is distinctly relevant; near-zero gap ⇒ either many good hits (fine)
// or uniformly mediocre ones (combine with other signals).
func Top1Top2Gap(hits []Hit) float64 {
	if len(hits) < 2 {
		if len(hits) == 1 {
			return hits[0].Score
		}
		return 0
	}
	return hits[0].Score - hits[1].Score
}

// RelativeDropoff (2b): top-1 vs the mean of the rest. Ratio >> 1 ⇒ the
// winner is a real outlier; ~1 ⇒ everything scored about the same, i.e. the
// retriever couldn't tell the candidates apart.
func RelativeDropoff(hits []Hit) float64 {
	if len(hits) < 2 {
		if len(hits) == 1 {
			return math.Inf(1)
		}
		return 0
	}
	mean := 0.0
	for _, h := range hits[1:] {
		mean += h.Score
	}
	mean /= float64(len(hits) - 1)
	if mean <= 0 {
		return math.Inf(1)
	}
	return hits[0].Score / mean
}

// SoftmaxEntropy (2c): how "peaked" is the score distribution? Softmax the
// scores (temperature sharpens/flattens), take Shannon entropy, normalise to
// [0,1] by dividing by log(K). ~0 ⇒ all mass on one hit (confident);
// ~1 ⇒ uniform (retriever is lost).
func SoftmaxEntropy(hits []Hit, temperature float64) float64 {
	if len(hits) < 2 {
		return 0
	}
	exps := make([]float64, len(hits))
	sum := 0.0
	for i, h := range hits {
		exps[i] = math.Exp(h.Score / temperature)
		sum += exps[i]
	}
	entropy := 0.0
	for _, e := range exps {
		p := e / sum
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	return entropy / math.Log(float64(len(hits)))
}

// ZScoreVsBackground (2d): is the top hit an outlier compared to how similar
// RANDOM chunks are to the query? Estimate mu/sigma from a random corpus
// sample. z > ~3 means the top hit is genuinely special, whatever the
// absolute cosine says.
func ZScoreVsBackground(topScore float64, backgroundScores []float64) float64 {
	mu := 0.0
	for _, s := range backgroundScores {
		mu += s
	}
	mu /= float64(len(backgroundScores))
	variance := 0.0
	for _, s := range backgroundScores {
		variance += (s - mu) * (s - mu)
	}
	sigma := math.Sqrt(variance / float64(len(backgroundScores)))
	if sigma == 0 {
		return 0
	}
	return (topScore - mu) / sigma
}

// CrossRetrieverAgreement (2e): Jaccard overlap of two independent top-K
// lists (e.g. dense + BM25). Two retrievers with different failure modes
// agreeing on the same chunks is strong evidence they found something real;
// zero overlap on an in-domain query is a red flag.
func CrossRetrieverAgreement(denseIDs, keywordIDs []string) float64 {
	a := map[string]bool{}
	for _, id := range denseIDs {
		a[id] = true
	}
	inter := 0
	union := map[string]bool{}
	for _, id := range denseIDs {
		union[id] = true
	}
	for _, id := range keywordIDs {
		if a[id] {
			inter++
		}
		union[id] = true
	}
	if len(union) == 0 {
		return 0
	}
	return float64(inter) / float64(len(union))
}

// ===========================================================================
// 3. LLM-BASED GROUNDEDNESS CHECKS (most robust, most expensive)
// ===========================================================================
// Ask a model to READ the evidence. This catches what vector math cannot:
// a chunk can be topically similar yet not actually answer the question
// ("mentions HNSW" vs "tells you how to tune HNSW"). Judges are injected as
// functions so the demo can use lexical mocks and production can use Claude.

type Answerability string

const (
	AnswerYes     Answerability = "YES"
	AnswerPartial Answerability = "PARTIAL"
	AnswerNo      Answerability = "NO"
)

// AnswerabilityCheck (3a) — pre-generation gate. One LLM call: "Can these
// passages answer this query?" The single highest-value LLM check: it runs
// BEFORE you spend tokens generating a wrong answer.
func AnswerabilityCheck(query string, hits []Hit, judge func(query string, passages []string) Answerability) Answerability {
	passages := make([]string, len(hits))
	for i, h := range hits {
		passages[i] = h.Text
	}
	return judge(query, passages)
}

// GradePassages (3b) — pointwise filter. Grade each passage 0–10 for "does it
// answer the query" and drop the ones below cutoff, so junk never reaches the
// generator's context. (Same mechanism as 3.6's pointwise LLM rerank, used
// here as a gate rather than a sort.)
func GradePassages(query string, hits []Hit, grader func(query, passage string) int, cutoff int) []Hit {
	out := make([]Hit, 0, len(hits))
	for _, h := range hits {
		if grader(query, h.Text) >= cutoff {
			out = append(out, h)
		}
	}
	return out
}

// GroundednessResult holds claim-level verification results.
type GroundednessResult struct {
	Supported   []string
	Unsupported []string
	Ratio       float64
}

// GroundednessCheck (3c) — post-generation, NLI-style. Split the draft answer
// into claims; for each, ask "is this claim supported by the evidence?"
// Unsupported claims are hallucination candidates: strip them, hedge, or
// regenerate.
func GroundednessCheck(answerClaims []string, evidence []Hit, entails func(claim string, evidence []string) bool) GroundednessResult {
	texts := make([]string, len(evidence))
	for i, h := range evidence {
		texts[i] = h.Text
	}
	var res GroundednessResult
	for _, claim := range answerClaims {
		if entails(claim, texts) {
			res.Supported = append(res.Supported, claim)
		} else {
			res.Unsupported = append(res.Unsupported, claim)
		}
	}
	if len(answerClaims) == 0 {
		res.Ratio = 1
	} else {
		res.Ratio = float64(len(res.Supported)) / float64(len(answerClaims))
	}
	return res
}

// Citation pairs a claim with the chunk it cites.
type Citation struct {
	Claim   string
	ChunkID string
}

// VerifyCitations (3d): for each "fact [chunk-id]" citation in the answer,
// check the cited chunk actually contains/entails the fact. Catches the
// classic failure of citing a real chunk for a made-up statement.
func VerifyCitations(citations []Citation, hits []Hit, entails func(claim string, evidence []string) bool) (valid, invalid int) {
	byID := map[string]Hit{}
	for _, h := range hits {
		byID[h.ID] = h
	}
	for _, c := range citations {
		if h, ok := byID[c.ChunkID]; ok && entails(c.Claim, []string{h.Text}) {
			valid++
		} else {
			invalid++
		}
	}
	return valid, invalid
}

// SelfConsistency (3e): sample the answer N times (temperature > 0); if the
// samples disagree, the model is guessing, whatever the retrieval scores
// said. Agreement is the fraction of samples matching the majority answer.
// N× generation cost — reserve for high-stakes answers.
func SelfConsistency(sampleAnswer func() string, n int) (majority string, agreement float64) {
	counts := map[string]int{}
	for i := 0; i < n; i++ {
		counts[sampleAnswer()]++
	}
	best := 0
	for a, c := range counts {
		if c > best {
			majority, best = a, c
		}
	}
	return majority, float64(best) / float64(n)
}

// ===========================================================================
// 4. THE COMPLETE CONFIDENCE PIPELINE
// ===========================================================================
// Compose the signals cheapest-first with early exits. Most queries decide at
// stage 1–2 for free; only the gray zone pays for LLM calls. Result:
// LLM-grade robustness at near-vector-math average cost.

// PipelineConfig holds the tunable cutoffs (calibrate on a dev set!).
type PipelineConfig struct {
	TauHard           float64 // stage 1: below this top-1 score, refuse immediately
	GapConfident      float64 // stage 2: gap ≥ this ⇒ skip LLM gate
	EntropyConfident  float64 // stage 2: entropy ≤ this ⇒ skip LLM gate
	EntropyHopeless   float64 // stage 2: entropy ≥ this AND weak top-1 ⇒ refuse
	GroundednessMin   float64 // stage 5: below this claim-support ratio ⇒ hedge
}

// PipelineDeps are the injected model calls.
type PipelineDeps struct {
	AnswerabilityJudge func(query string, passages []string) Answerability
	Generate           func(query string, hits []Hit) (answer string, claims []string)
	Entails            func(claim string, evidence []string) bool
}

// PipelineResult reports the verdict and a human-readable trace.
type PipelineResult struct {
	Verdict   Verdict
	Answer    string
	DecidedAt string
	Trace     []string
}

// ConfidencePipeline runs the full cheapest-first cascade.
func ConfidencePipeline(query string, hits []Hit, cfg PipelineConfig, deps PipelineDeps) PipelineResult {
	var trace []string

	// --- Stage 1: absolute floor (free, instant) --------------------------
	top1 := 0.0
	if len(hits) > 0 {
		top1 = hits[0].Score
	}
	trace = append(trace, fmt.Sprintf("stage 1 [absolute]      top-1=%.3f vs τ_hard=%.3f", top1, cfg.TauHard))
	if !Top1Floor(hits, cfg.TauHard) {
		return PipelineResult{Verdict: VerdictRefuse, DecidedAt: "stage 1: absolute floor", Trace: trace}
	}

	// --- Stage 2: distributional shape (free) ------------------------------
	gap := Top1Top2Gap(hits)
	entropy := SoftmaxEntropy(hits, 0.1)
	trace = append(trace, fmt.Sprintf("stage 2 [distributional] gap=%.3f entropy=%.3f", gap, entropy))
	needLLMGate := true
	if gap >= cfg.GapConfident || entropy <= cfg.EntropyConfident {
		trace = append(trace, "stage 2: clear winner — skipping LLM gate")
		needLLMGate = false // confident: go straight to generation
	} else if entropy >= cfg.EntropyHopeless && top1 < cfg.TauHard*1.5 {
		return PipelineResult{Verdict: VerdictRefuse, DecidedAt: "stage 2: flat distribution + weak scores", Trace: trace}
	}

	// --- Stage 3: LLM answerability gate (only for the gray zone) ----------
	hedging := false
	if needLLMGate {
		verdict := AnswerabilityCheck(query, hits, deps.AnswerabilityJudge)
		trace = append(trace, fmt.Sprintf("stage 3 [LLM gate]      answerability=%s", verdict))
		if verdict == AnswerNo {
			return PipelineResult{Verdict: VerdictRefuse, DecidedAt: "stage 3: LLM answerability", Trace: trace}
		}
		hedging = verdict == AnswerPartial
	}

	// --- Stage 4: generate with citations -----------------------------------
	answer, claims := deps.Generate(query, hits)
	trace = append(trace, fmt.Sprintf("stage 4 [generate]      %d claim(s)", len(claims)))

	// --- Stage 5: groundedness verification ---------------------------------
	g := GroundednessCheck(claims, hits, deps.Entails)
	trace = append(trace, fmt.Sprintf("stage 5 [groundedness]  %d/%d claims supported (ratio=%.2f)", len(g.Supported), len(claims), g.Ratio))
	if g.Ratio < cfg.GroundednessMin {
		hedging = true
	}

	if hedging {
		return PipelineResult{Verdict: VerdictHedge, Answer: answer, DecidedAt: "stage 5: partial support → hedge", Trace: trace}
	}
	return PipelineResult{Verdict: VerdictAnswer, Answer: answer, DecidedAt: "stage 5: fully grounded", Trace: trace}
}
