// main.go — runnable demo for Retrieval Confidence.
//
//	go run .
//
// No API key required. Mocks:
//   - mockEmbed  : 6-dim concept vector (embedding stand-in)
//   - mock judges: lexical overlap heuristics (LLM stand-ins)
//
// Three queries against a vector-DB knowledge base:
//   Q1 answerable   — corpus covers it well
//   Q2 ambiguous    — corpus only partially covers it
//   Q3 unanswerable — off-corpus (cooking); retrieval still "returns" chunks!
//
// For each family of signals we print what it reads on all three queries,
// then run the Complete Confidence Pipeline end to end.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// SAMPLE CORPUS — same vector-database theme as 3.5/3.6
// ---------------------------------------------------------------------------

type doc struct {
	id   string
	text string
}

var corpus = []doc{
	{"hnsw-tuning", "To make HNSW faster without hurting recall, increase efSearch gradually and tune M; measure recall on a held-out query set."},
	{"hnsw-basics", "HNSW is a graph-based index for approximate nearest neighbour search with strong recall/latency trade-offs."},
	{"ivf-index", "IVF indexes partition vectors into clusters; probe more lists for better recall at higher latency."},
	{"pgvector-ops", "pgvector supports cosine, L2 and inner-product operators; build an HNSW index for production workloads."},
	{"embedding-models", "Choosing an embedding model: balance dimensionality, cost and retrieval quality; re-embed when you switch models."},
	{"chunking", "Chunking strategy affects retrieval: overlapping windows help recall, semantic chunking keeps ideas intact."},
}

// ---------------------------------------------------------------------------
// MOCK EMBEDDINGS — 6 concept dimensions
// [hnsw, index/search, recall/latency, postgres, embeddings, food]
// ---------------------------------------------------------------------------

var concepts = []struct {
	re  *regexp.Regexp
	dim int
}{
	{regexp.MustCompile(`(?i)hnsw|efsearch|^m$|graph`), 0},
	{regexp.MustCompile(`(?i)index|search|ivf|nearest|probe|lists`), 1},
	{regexp.MustCompile(`(?i)recall|latency|faster|speed|trade`), 2},
	{regexp.MustCompile(`(?i)pgvector|postgres|operator`), 3},
	{regexp.MustCompile(`(?i)embed|model|dimension|chunk`), 4},
	{regexp.MustCompile(`(?i)cook|pasta|carbonara|recipe|egg|cheese`), 5},
}

var wordSplit = regexp.MustCompile(`\W+`)

func mockEmbed(text string) []float64 {
	v := make([]float64, 6)
	for _, w := range wordSplit.Split(strings.ToLower(text), -1) {
		for _, c := range concepts {
			if c.re.MatchString(w) {
				v[c.dim]++
			}
		}
	}
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		norm = 1
	}
	for i := range v {
		v[i] /= norm
	}
	return v
}

func cosine(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func retrieve(query string, k int) []Hit {
	qv := mockEmbed(query)
	hits := make([]Hit, 0, len(corpus))
	for _, d := range corpus {
		hits = append(hits, Hit{ID: d.id, Text: d.text, Score: cosine(qv, mockEmbed(d.text))})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if k < len(hits) {
		hits = hits[:k]
	}
	return hits
}

// keywordRetrieve is a BM25-ish mock: rank by term overlap.
func keywordRetrieve(query string, k int) []string {
	qTerms := map[string]bool{}
	for _, w := range wordSplit.Split(strings.ToLower(query), -1) {
		if len(w) > 3 {
			qTerms[w] = true
		}
	}
	type scored struct {
		id      string
		overlap int
	}
	var s []scored
	for _, d := range corpus {
		n := 0
		for _, w := range wordSplit.Split(strings.ToLower(d.text), -1) {
			if qTerms[w] {
				n++
			}
		}
		if n > 0 {
			s = append(s, scored{d.id, n})
		}
	}
	sort.Slice(s, func(i, j int) bool { return s[i].overlap > s[j].overlap })
	if k < len(s) {
		s = s[:k]
	}
	ids := make([]string, len(s))
	for i, x := range s {
		ids[i] = x.id
	}
	return ids
}

// ---------------------------------------------------------------------------
// MOCK LLM JUDGES — lexical overlap standing in for Claude
// ---------------------------------------------------------------------------

func termOverlap(a, b string) float64 {
	ta := map[string]bool{}
	for _, w := range wordSplit.Split(strings.ToLower(a), -1) {
		if len(w) > 3 {
			ta[w] = true
		}
	}
	tb := map[string]bool{}
	for _, w := range wordSplit.Split(strings.ToLower(b), -1) {
		if len(w) > 3 {
			tb[w] = true
		}
	}
	if len(ta) == 0 {
		return 0
	}
	n := 0
	for t := range ta {
		if tb[t] {
			n++
		}
	}
	return float64(n) / float64(len(ta))
}

func mockAnswerabilityJudge(query string, passages []string) Answerability {
	best := 0.0
	for _, p := range passages {
		if o := termOverlap(query, p); o > best {
			best = o
		}
	}
	switch {
	case best >= 0.4:
		return AnswerYes
	case best >= 0.15:
		return AnswerPartial
	default:
		return AnswerNo
	}
}

func mockGrader(query, passage string) int {
	return int(math.Round(termOverlap(query, passage) * 10))
}

func mockEntails(claim string, evidence []string) bool {
	for _, e := range evidence {
		if termOverlap(claim, e) >= 0.5 {
			return true
		}
	}
	return false
}

// mockGenerate is an extractive "generator": answers with the top passage,
// claims = its sentences.
func mockGenerate(_ string, hits []Hit) (string, []string) {
	answer := hits[0].Text
	var claims []string
	for _, c := range strings.FieldsFunc(answer, func(r rune) bool { return r == '.' || r == ';' }) {
		if c = strings.TrimSpace(c); c != "" {
			claims = append(claims, c)
		}
	}
	return answer, claims
}

// ---------------------------------------------------------------------------
// DEMO
// ---------------------------------------------------------------------------

func section(title string) {
	line := strings.Repeat("=", 75)
	fmt.Printf("\n%s\n%s\n%s\n", line, title, line)
}

func main() {
	queries := []struct{ label, q string }{
		{"Q1 answerable  ", "How do I make HNSW search faster without hurting recall?"},
		{"Q2 ambiguous   ", "Should I re-embed my chunks when switching from IVF to HNSW in Postgres?"},
		{"Q3 unanswerable", "What is the best recipe for pasta carbonara with eggs and cheese?"},
	}

	// --- 1. Absolute thresholding -----------------------------------------
	section("1. ABSOLUTE THRESHOLDING (crude, corpus-specific)")

	// 1d: calibrate τ from known-irrelevant pairs instead of guessing.
	var irrelevant []float64
	for _, q := range []string{"carbonara recipe", "football scores", "weather tomorrow"} {
		for _, h := range retrieve(q, len(corpus)) {
			irrelevant = append(irrelevant, h.Score)
		}
	}
	tau := math.Max(0.15, CalibrateThreshold(irrelevant, 0.95))
	fmt.Printf("calibrated τ from irrelevant-pair scores (95th pct): %.3f\n\n", tau)

	for _, qq := range queries {
		hits := retrieve(qq.q, 4)
		fmt.Printf("%s top-1=%.3f (%s)\n", qq.label, hits[0].Score, hits[0].ID)
		fmt.Printf("   1a similarity floor : %d/%d hits survive τ=%.2f\n", len(SimilarityFloor(hits, tau)), len(hits), tau)
		gate := "REFUSE"
		if Top1Floor(hits, tau) {
			gate = "proceed"
		}
		fmt.Printf("   1b top-1 floor      : %s\n", gate)
		fmt.Printf("   1c count ≥2 above τ : %v\n", CountAboveThreshold(hits, tau, 2))
		fmt.Printf("   1e reranker floor   : %d hits pass LLM-grade ≥5\n", len(GradePassages(qq.q, hits, mockGrader, 5)))
	}

	// --- 2. Relative / distributional signals ------------------------------
	section("2. RELATIVE / DISTRIBUTIONAL SIGNALS (more robust)")

	for _, qq := range queries {
		hits := retrieve(qq.q, 4)
		qv := mockEmbed(qq.q)
		var background []float64
		for _, d := range corpus {
			background = append(background, cosine(qv, mockEmbed(d.text)))
		}
		denseIDs := make([]string, len(hits))
		for i, h := range hits {
			denseIDs[i] = h.ID
		}
		fmt.Printf("%s\n", qq.label)
		fmt.Printf("   2a top1-top2 gap    : %.3f\n", Top1Top2Gap(hits))
		fmt.Printf("   2b relative dropoff : %.2fx\n", RelativeDropoff(hits))
		fmt.Printf("   2c softmax entropy  : %.3f  (0=peaked, 1=uniform)\n", SoftmaxEntropy(hits, 0.1))
		fmt.Printf("   2d z-score vs corpus: %.2f\n", ZScoreVsBackground(hits[0].Score, background))
		fmt.Printf("   2e dense∩keyword    : %.2f Jaccard\n", CrossRetrieverAgreement(denseIDs, keywordRetrieve(qq.q, 4)))
	}

	// --- 3. LLM-based groundedness checks -----------------------------------
	section("3. LLM-BASED GROUNDEDNESS CHECKS (most robust, most expensive)")

	for _, qq := range queries {
		hits := retrieve(qq.q, 4)
		gate := AnswerabilityCheck(qq.q, hits, mockAnswerabilityJudge)
		fmt.Printf("%s\n", qq.label)
		fmt.Printf("   3a answerability    : %s\n", gate)
		grades := make([]string, len(hits))
		for i, h := range hits {
			grades[i] = fmt.Sprintf("%d", mockGrader(qq.q, h.Text))
		}
		fmt.Printf("   3b passage grades   : [%s] → %d kept\n", strings.Join(grades, ", "), len(GradePassages(qq.q, hits, mockGrader, 5)))
		_, claims := mockGenerate(qq.q, hits)
		g := GroundednessCheck(claims, hits, mockEntails)
		fmt.Printf("   3c groundedness     : %d/%d claims supported\n", len(g.Supported), len(claims))
		pAgree := 0.4
		if gate == AnswerYes {
			pAgree = 0.95
		}
		_, agreement := SelfConsistency(func() string {
			if rand.Float64() < pAgree {
				return hits[0].ID
			}
			return hits[1].ID
		}, 5)
		fmt.Printf("   3e self-consistency : %.0f%% of 5 samples agree\n", agreement*100)
	}

	// --- 4. The Complete Confidence Pipeline --------------------------------
	section("4. THE COMPLETE CONFIDENCE PIPELINE (cheapest-first, early exits)")

	cfg := PipelineConfig{
		TauHard:          tau,
		GapConfident:     0.15,
		EntropyConfident: 0.5,
		EntropyHopeless:  0.9,
		GroundednessMin:  0.7,
	}
	deps := PipelineDeps{
		AnswerabilityJudge: mockAnswerabilityJudge,
		Generate:           mockGenerate,
		Entails:            mockEntails,
	}

	for _, qq := range queries {
		res := ConfidencePipeline(qq.q, retrieve(qq.q, 4), cfg, deps)
		fmt.Printf("\n%s %q\n", qq.label, qq.q)
		for _, line := range res.Trace {
			fmt.Printf("   %s\n", line)
		}
		fmt.Printf("   → %s  (decided at %s)\n", res.Verdict, res.DecidedAt)
		if res.Answer != "" {
			fmt.Printf("   answer: %s\n", res.Answer)
		}
	}
}
