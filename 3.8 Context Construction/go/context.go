// context.go — Context construction for RAG.
//
// Retrieval hands you a ranked list of chunks. The model sees a *string*.
// Everything between those two facts is context construction:
//
//  1. Context Packing      — fit maximum relevant content in the token budget
//  2. Context Ordering     — best evidence at the top AND bottom, not the middle
//  3. Context Compression  — summarise / trim chunks before injection
//  4. Citation formatting  — label evidence so the answer can point back at it
//     + the Complete Context Construction Pipeline that composes all four.
//
// Everything is model-agnostic: token counting and LLM summarisation come in
// as injected functions, so the mocks in main.go can be swapped for a real
// tokenizer and a real Claude call without touching this file.
package main

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

// Chunk is a retrieved chunk plus the provenance a citation will need.
type Chunk struct {
	ID      string
	Text    string
	Score   float64 // relevance from retrieval/reranking (higher = better)
	Source  string  // file path, URL or document title — shown in the citation
	Loc     string  // location inside the source ("L20-L48", "§3.2", "p.7")
	Ordinal int     // position of this chunk inside its source document
}

// TokenCounter is a rough token counter. Swap for tiktoken / count-tokens API.
type TokenCounter func(string) int

// Summarizer is a query-focused abstractive compressor (an LLM call in prod).
type Summarizer func(text, query string) string

func sortByScore(chunks []Chunk) []Chunk {
	out := append([]Chunk(nil), chunks...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}

// ===========================================================================
// 1. CONTEXT PACKING — fitting maximum relevant content within token limits
// ===========================================================================
// The context window is a budget, not a container: the prompt scaffolding, the
// query, the conversation history and the *reserved output* all spend from the
// same pool. Packing is the knapsack problem — maximise retrieved value under
// a hard token ceiling.

// BudgetSpec describes everything competing for the window.
type BudgetSpec struct {
	ContextWindow int // model's total window
	SystemTokens  int // system prompt + citation instructions
	QueryTokens   int // the user question (+ chat history)
	AnswerReserve int // room the model needs to WRITE the answer
	SafetyMargin  int // tokenizer disagreement, formatting overhead
}

// EvidenceBudget (1a) works out how many tokens are actually left for evidence.
// The single most common production bug in RAG is packing to the model's
// context size and then getting truncated because the answer had nowhere to
// go. Always subtract the reservations first.
func EvidenceBudget(s BudgetSpec) int {
	margin := s.SafetyMargin
	if margin == 0 {
		margin = 128
	}
	b := s.ContextWindow - s.SystemTokens - s.QueryTokens - s.AnswerReserve - margin
	if b < 0 {
		return 0
	}
	return b
}

// PackGreedy (1b) takes chunks in relevance order while they fit.
// Note the `continue` (not `break`): a single oversized chunk must not stop
// three smaller, still-relevant chunks from getting in.
func PackGreedy(chunks []Chunk, budget int, count TokenCounter) []Chunk {
	out := []Chunk{}
	used := 0
	for _, c := range sortByScore(chunks) {
		t := count(c.Text)
		if used+t > budget {
			continue // skip, don't stop
		}
		out = append(out, c)
		used += t
	}
	return out
}

// PackByDensity (1c) is the knapsack view: rank by value *per token*
// (score / tokens) instead of raw score, so a 40-token chunk scoring 0.7 beats
// a 900-token chunk scoring 0.8. This fits more distinct evidence in the same
// budget; the trade-off is that it favours short chunks, so keep the top hits
// pinned by relevance (pinTop) or you can lose the best answer.
func PackByDensity(chunks []Chunk, budget int, count TokenCounter, pinTop int) []Chunk {
	ranked := sortByScore(chunks)
	if pinTop > len(ranked) {
		pinTop = len(ranked)
	}
	out := []Chunk{}
	used := 0
	for _, c := range ranked[:pinTop] {
		if t := count(c.Text); used+t <= budget {
			out = append(out, c)
			used += t
		}
	}
	rest := append([]Chunk(nil), ranked[pinTop:]...)
	sort.SliceStable(rest, func(i, j int) bool {
		di := rest[i].Score / math.Max(1, float64(count(rest[i].Text)))
		dj := rest[j].Score / math.Max(1, float64(count(rest[j].Text)))
		return di > dj
	})
	for _, c := range rest {
		t := count(c.Text)
		if used+t > budget {
			continue
		}
		out = append(out, c)
		used += t
	}
	return out
}

// Dedupe (1d) removes near-duplicates — the cheapest way to buy budget.
// Overlapping windows (3.2) and multi-query retrieval (3.3) routinely return
// the same paragraph two or three times. Duplicated evidence also biases the
// model: it reads repetition as importance. Jaccard over word sets is a crude
// but effective stand-in for a MinHash/SimHash de-duplicator.
func Dedupe(chunks []Chunk, threshold float64) []Chunk {
	kept := []Chunk{}
	keptWords := []map[string]bool{}
	for _, c := range sortByScore(chunks) {
		w := wordSet(c.Text)
		dup := false
		for _, kw := range keptWords {
			if jaccard(w, kw) >= threshold {
				dup = true
				break
			}
		}
		if !dup {
			kept = append(kept, c)
			keptWords = append(keptWords, w)
		}
	}
	return kept
}

var nonWord = regexp.MustCompile(`\W+`)

func wordSet(s string) map[string]bool {
	m := map[string]bool{}
	for _, w := range nonWord.Split(strings.ToLower(s), -1) {
		if w != "" {
			m[w] = true
		}
	}
	return m
}

func jaccard(a, b map[string]bool) float64 {
	inter := 0
	for x := range a {
		if b[x] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// CapPerSource (1e) never lets one file eat the whole window. Ten chunks from
// one file answer one question well and every other question badly. Capping
// per source forces breadth, which matters most for "how does X interact with
// Y" questions spanning several files.
func CapPerSource(chunks []Chunk, maxPerSource int) []Chunk {
	seen := map[string]int{}
	out := []Chunk{}
	for _, c := range sortByScore(chunks) {
		if seen[c.Source] >= maxPerSource {
			continue
		}
		seen[c.Source]++
		out = append(out, c)
	}
	return out
}

// PackTiered (1f) gives full text to the top tier and compresses the tail.
// Precision where it matters, recall where it's cheap: the best fullTier
// chunks go in verbatim (so they can be quoted exactly), everything after is
// squeezed through compress until the budget is exhausted.
func PackTiered(chunks []Chunk, budget int, count TokenCounter, fullTier int, compress func(Chunk) string) []Chunk {
	out := []Chunk{}
	used := 0
	for i, c := range sortByScore(chunks) {
		text := c.Text
		if i >= fullTier {
			text = compress(c)
		}
		t := count(text)
		if used+t > budget {
			continue
		}
		c.Text = text
		out = append(out, c)
		used += t
	}
	return out
}

// TotalTokens is the tokens a chunk list occupies (text only — markup adds more).
func TotalTokens(chunks []Chunk, count TokenCounter) int {
	n := 0
	for _, c := range chunks {
		n += count(c.Text)
	}
	return n
}

// ===========================================================================
// 2. CONTEXT ORDERING — highest relevance at top and bottom, not the middle
// ===========================================================================
// "Lost in the middle" (Liu et al., 2023): with the same evidence in the
// window, accuracy is highest when the answer sits at the START or the END of
// the context and drops sharply when it is buried in the middle — a U-shaped
// curve. So the order you emit chunks in is a retrieval decision, not a
// cosmetic one, and simply sorting by score descending wastes the strong tail
// position on your weakest evidence.

// OrderByRelevance (2a) is the naive baseline: best evidence gets the strong
// head position, worst evidence gets the *other* strong position.
func OrderByRelevance(chunks []Chunk) []Chunk { return sortByScore(chunks) }

// OrderBookend (2b) is the lost-in-the-middle fix. Alternate outwards-in:
// rank 1 first, rank 2 last, rank 3 second, rank 4 second-to-last… so the
// strongest chunks occupy both attention peaks and the weakest sink into the
// middle where attention is cheapest anyway. Default above ~5 chunks.
func OrderBookend(chunks []Chunk) []Chunk {
	ranked := sortByScore(chunks)
	head, tail := []Chunk{}, []Chunk{}
	for i, c := range ranked {
		if i%2 == 0 {
			head = append(head, c)
		} else {
			tail = append([]Chunk{c}, tail...) // unshift
		}
	}
	return append(head, tail...)
}

// groupBySource returns per-source groups in reading order, groups themselves
// ranked by their best chunk.
func groupBySource(chunks []Chunk) [][]Chunk {
	order := []string{}
	groups := map[string][]Chunk{}
	for _, c := range chunks {
		if _, ok := groups[c.Source]; !ok {
			order = append(order, c.Source)
		}
		groups[c.Source] = append(groups[c.Source], c)
	}
	out := [][]Chunk{}
	for _, s := range order {
		g := groups[s]
		sort.SliceStable(g, func(i, j int) bool { return g[i].Ordinal < g[j].Ordinal })
		out = append(out, g)
	}
	best := func(g []Chunk) float64 {
		m := math.Inf(-1)
		for _, c := range g {
			m = math.Max(m, c.Score)
		}
		return m
	}
	sort.SliceStable(out, func(i, j int) bool { return best(out[i]) > best(out[j]) })
	return out
}

// OrderByDocument (2c) groups chunks by source and puts them back in reading
// order inside each group. Interleaved fragments of three files read as noise;
// contiguous runs read as a document. Sources are ordered by their best chunk,
// so relevance still drives the top-level layout. Use this when chunks are
// narrative/sequential (code files, tutorials).
func OrderByDocument(chunks []Chunk) []Chunk {
	out := []Chunk{}
	for _, g := range groupBySource(chunks) {
		out = append(out, g...)
	}
	return out
}

// OrderGroupedBookend (2d) is the production compromise: keep each source's
// chunks contiguous (coherence) but place whole *groups* at the attention
// peaks (recency/primacy). Best of 2b and 2c.
func OrderGroupedBookend(chunks []Chunk) []Chunk {
	groups := groupBySource(chunks)
	head, tail := [][]Chunk{}, [][]Chunk{}
	for i, g := range groups {
		if i%2 == 0 {
			head = append(head, g)
		} else {
			tail = append([][]Chunk{g}, tail...)
		}
	}
	out := []Chunk{}
	for _, g := range append(head, tail...) {
		out = append(out, g...)
	}
	return out
}

// Layout is the three prompt parts in emission order.
type Layout struct{ Pre, Context, Post string }

// QueryAdjacentLayout (2e): where the *question* goes matters as much as where
// the evidence goes. Repeating the query immediately after the context (so it
// occupies the final, strongest position) reliably beats asking once before a
// long context.
func QueryAdjacentLayout(query, context, instructions string) Layout {
	return Layout{
		Pre:     fmt.Sprintf("%s\n\nQuestion: %s", instructions, query),
		Context: context,
		Post: fmt.Sprintf("Question (restated): %s\nAnswer using only the context above, with citations.",
			query),
	}
}

// ===========================================================================
// 3. CONTEXT COMPRESSION — summarising or trimming chunks before injection
// ===========================================================================
// Compression trades fidelity for budget. The rule that keeps it safe: never
// compress the evidence you expect to quote. Compress the supporting cast so
// the star witness fits verbatim.

var sentenceEnd = regexp.MustCompile(`([.!?;])\s+`)

func splitSentences(text string) []string {
	parts := sentenceEnd.Split(text, -1)
	marks := sentenceEnd.FindAllStringSubmatch(text, -1)
	out := []string{}
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if i < len(marks) {
			p += marks[i][1] // put the punctuation back
		}
		out = append(out, p)
	}
	return out
}

// ExtractiveCompress (3a) keeps only the sentences of a chunk that overlap the
// query. Free, lossless at the sentence level, and it preserves the exact
// wording, so the result is still quotable. Start here.
func ExtractiveCompress(text, query string, maxSentences int) string {
	q := wordSet(query)
	type scored struct {
		s     string
		i     int
		score float64
	}
	sentences := splitSentences(text)
	all := make([]scored, 0, len(sentences))
	for i, s := range sentences {
		all = append(all, scored{s, i, sentenceScore(s, q)})
	}
	sort.SliceStable(all, func(i, j int) bool { return all[i].score > all[j].score })
	if maxSentences < len(all) {
		all = all[:maxSentences]
	}
	sort.SliceStable(all, func(i, j int) bool { return all[i].i < all[j].i }) // reading order
	keep := make([]string, 0, len(all))
	for _, s := range all {
		keep = append(keep, s.s)
	}
	return strings.Join(keep, " ")
}

func sentenceScore(sentence string, q map[string]bool) float64 {
	words := []string{}
	for _, w := range nonWord.Split(strings.ToLower(sentence), -1) {
		if w != "" {
			words = append(words, w)
		}
	}
	if len(words) == 0 {
		return 0
	}
	hits := 0
	for _, w := range words {
		if q[w] {
			hits++
		}
	}
	return float64(hits) / math.Sqrt(float64(len(words))) // length-normalised
}

// TruncateToTokens (3b) is a hard cap that never cuts mid-sentence. A chunk
// severed mid-clause invites the model to complete the thought itself, which
// is exactly how truncation turns into hallucination.
func TruncateToTokens(text string, maxTokens int, count TokenCounter) string {
	if count(text) <= maxTokens {
		return text
	}
	out := []string{}
	used := 0
	for _, s := range splitSentences(text) {
		t := count(s)
		if used+t > maxTokens {
			break
		}
		out = append(out, s)
		used += t
	}
	if len(out) == 0 {
		n := maxTokens * 4
		if n > len(text) {
			n = len(text)
		}
		return text[:n] + " […]"
	}
	return strings.Join(out, " ") + " […]"
}

// PruneRedundantSentences (3c) dedupes at *sentence* level across the whole
// context. Near-duplicate chunks survive Dedupe (they differ overall) while
// still repeating the same key sentence three times; this catches that.
func PruneRedundantSentences(chunks []Chunk) []Chunk {
	seen := map[string]bool{}
	out := []Chunk{}
	for _, c := range chunks {
		keep := []string{}
		for _, s := range splitSentences(c.Text) {
			k := strings.TrimSpace(nonWord.ReplaceAllString(strings.ToLower(s), " "))
			if k == "" || seen[k] {
				continue
			}
			seen[k] = true
			keep = append(keep, s)
		}
		c.Text = strings.Join(keep, " ")
		if strings.TrimSpace(c.Text) != "" {
			out = append(out, c)
		}
	}
	return out
}

// AbstractiveCompress (3d) produces a query-focused summary of a chunk.
// Highest compression ratio, highest cost, and the only lossy option: the
// output is paraphrase, so it can no longer be quoted verbatim. Mark such
// chunks as summaries in the prompt so the model doesn't invent quotes.
func AbstractiveCompress(c Chunk, query string, summarize Summarizer) Chunk {
	c.Text = summarize(c.Text, query)
	return c
}

// CascadeResult reports what the cascade did and which chunks it paraphrased.
type CascadeResult struct {
	Chunks     []Chunk
	Stages     []string
	Summarised map[string]bool // ids rewritten by the LLM (no longer quotable)
}

// CompressionCascade (3e) only compresses when it must, cheapest first:
// dedupe → sentence pruning → extractive on the tail → LLM summaries on the
// tail → drop. Exits the moment the list fits, so a query whose evidence
// already fits pays nothing. protectTop chunks are never compressed — they are
// the ones the answer will quote.
func CompressionCascade(chunks []Chunk, query string, budget int, count TokenCounter,
	summarize Summarizer, protectTop int) CascadeResult {

	res := CascadeResult{Summarised: map[string]bool{}}
	cur := OrderByRelevance(chunks)
	fits := func() bool { return TotalTokens(cur, count) <= budget }
	stage := func(label string) {
		res.Stages = append(res.Stages, fmt.Sprintf("%s → %d tok", label, TotalTokens(cur, count)))
	}

	res.Stages = append(res.Stages,
		fmt.Sprintf("start: %d tok vs budget %d", TotalTokens(cur, count), budget))
	if fits() {
		res.Chunks = cur
		res.Stages = append(res.Stages, "fits — no compression")
		return res
	}

	cur = Dedupe(cur, 0.8)
	stage("dedupe")
	if fits() {
		res.Chunks = cur
		return res
	}

	cur = PruneRedundantSentences(cur)
	stage("prune redundant sentences")
	if fits() {
		res.Chunks = cur
		return res
	}

	for i := range cur {
		if i >= protectTop {
			cur[i].Text = ExtractiveCompress(cur[i].Text, query, 2)
		}
	}
	stage("extractive (tail)")
	if fits() {
		res.Chunks = cur
		return res
	}

	for i := range cur {
		if i >= protectTop {
			res.Summarised[cur[i].ID] = true
			cur[i] = AbstractiveCompress(cur[i], query, summarize)
		}
	}
	stage("LLM summaries (tail)")
	if fits() {
		res.Chunks = cur
		return res
	}

	cur = PackGreedy(cur, budget, count)
	stage("still over — drop lowest-value chunks")
	res.Chunks = cur
	return res
}

// ===========================================================================
// 4. CITATION-READY FORMATTING (sets up 3.9)
// ===========================================================================
// A model can only cite what you gave it a *handle* for. Formatting decides
// whether "according to the docs" or "[3] auth/session.go:L20-L48" is even
// expressible. Three requirements:
//
//	(i)   a stable, short marker per block ([1], [2], …)
//	(ii)  provenance next to the marker (source + location), so the answer's
//	      citation resolves back to a file/line without a lookup table
//	(iii) hard delimiters, so the model can tell evidence from instructions —
//	      and so injected text inside a chunk can't impersonate the prompt.

// FormattedContext is the prompt block plus the marker → chunk map used to
// verify and render citations after generation.
type FormattedContext struct {
	Text      string
	Citations map[int]Chunk
	Tokens    int
}

// FormatOptions controls the rendering of the evidence block.
type FormatOptions struct {
	// Style "xml" fences evidence off from instructions; Claude follows tags
	// well and they survive chunks that themselves contain markdown/code.
	Style         string          // "xml" (default) | "markdown"
	IncludeScores bool            // useful for debugging, noise for the model
	SummaryIDs    map[string]bool // LLM paraphrases — cite, never quote
	ExcerptIDs    map[string]bool // trimmed but verbatim — safe to quote
}

// FormatForCitation renders chunks as numbered, provenance-tagged blocks.
func FormatForCitation(chunks []Chunk, count TokenCounter, opts FormatOptions) FormattedContext {
	style := opts.Style
	if style == "" {
		style = "xml"
	}
	citations := map[int]Chunk{}
	blocks := make([]string, 0, len(chunks))
	for i, c := range chunks {
		n := i + 1
		citations[n] = c
		loc := ""
		if c.Loc != "" {
			loc = ":" + c.Loc
		}
		kind := ""
		switch {
		case opts.SummaryIDs[c.ID]:
			kind = "summary"
		case opts.ExcerptIDs[c.ID]:
			kind = "excerpt"
		}
		if style == "markdown" {
			suffix := ""
			if kind != "" {
				suffix = fmt.Sprintf(" (%s)", kind)
			}
			blocks = append(blocks, fmt.Sprintf("[%d] %s%s%s\n%s", n, c.Source, loc, suffix, c.Text))
			continue
		}
		kindAttr := ""
		if kind != "" {
			kindAttr = fmt.Sprintf(" kind=%q", kind)
		}
		score := ""
		if opts.IncludeScores {
			score = fmt.Sprintf(" score=%q", fmt.Sprintf("%.3f", c.Score))
		}
		blocks = append(blocks, fmt.Sprintf("<source id=%q path=%q%s%s>\n%s\n</source>",
			fmt.Sprint(n), c.Source+loc, kindAttr, score, c.Text))
	}
	text := strings.Join(blocks, "\n\n---\n\n")
	if style == "xml" {
		text = "<context>\n" + strings.Join(blocks, "\n") + "\n</context>"
	}
	return FormattedContext{Text: text, Citations: citations, Tokens: count(text)}
}

// CitationInstructions is the instruction half of the contract. Markers are
// worthless unless the system prompt tells the model exactly how to spend them
// — and what to do when the context does not contain the answer (see 3.7).
func CitationInstructions() string {
	return strings.Join([]string{
		"Answer the question using ONLY the numbered sources in <context>.",
		"After every sentence that uses a source, cite it inline as [n] — e.g. “efSearch trades latency for recall [2].”",
		"Cite multiple sources as [1][3] when a sentence draws on several.",
		`Sources marked kind="excerpt" are shortened but verbatim — safe to quote.`,
		`Sources marked kind="summary" are LLM paraphrases: cite them, but never quote them verbatim.`,
		`If the context does not contain the answer, reply exactly: "I don't know based on the provided sources."`,
		"Never cite a source number that does not appear in <context>.",
	}, "\n")
}

var citationMarker = regexp.MustCompile(`\[(\d+)\]`)

// ResolveCitations maps the [n] markers a model emitted back to real chunks —
// the bridge into 3.9 Citations, where these get verified and rendered as
// links. Invalid markers are hallucinated citations (3.7's groundedness
// problem showing up in a form you can detect with a regex).
func ResolveCitations(answer string, citations map[int]Chunk) (used []Chunk, usedNums []int, invalid []int) {
	seen := map[int]bool{}
	for _, m := range citationMarker.FindAllStringSubmatch(answer, -1) {
		var n int
		fmt.Sscanf(m[1], "%d", &n)
		if seen[n] {
			continue
		}
		seen[n] = true
		if c, ok := citations[n]; ok {
			used = append(used, c)
			usedNums = append(usedNums, n)
		} else {
			invalid = append(invalid, n)
		}
	}
	return
}

// ===========================================================================
// 5. THE COMPLETE CONTEXT CONSTRUCTION PIPELINE
// ===========================================================================
// Order of operations matters, and it is not the order the concepts are
// taught in:
//
//	dedupe → cap per source → compress to fit → pack under budget
//	       → order for attention → format with citation markers
//
// Compress BEFORE packing (compression changes what fits), pack BEFORE
// ordering (ordering is meaningless on chunks that get dropped), and format
// LAST (formatting overhead must be inside the budget you measured).

// BuildConfig is the knob panel for the pipeline.
type BuildConfig struct {
	ContextWindow   int
	SystemTokens    int
	AnswerReserve   int
	MaxPerSource    int
	DedupeThreshold float64
	ProtectTop      int    // top-N chunks never compressed (quotable evidence)
	Ordering        string // "relevance" | "bookend" | "document" | "grouped-bookend"
	Style           string // "xml" | "markdown"
}

// BuildResult is the finished prompt plus everything needed to audit it.
type BuildResult struct {
	Prompt  string // full prompt, ready to send
	Context FormattedContext
	Chunks  []Chunk // exactly what made it in, in emission order
	Trace   []string
}

// BuildContext runs the whole pipeline for one query.
func BuildContext(query string, hits []Chunk, cfg BuildConfig,
	count TokenCounter, summarize Summarizer) BuildResult {

	trace := []string{}
	instructions := CitationInstructions()

	// --- Step 0: budget -----------------------------------------------------
	budget := EvidenceBudget(BudgetSpec{
		ContextWindow: cfg.ContextWindow,
		SystemTokens:  cfg.SystemTokens + count(instructions),
		QueryTokens:   count(query) * 2, // the query appears twice (2e)
		AnswerReserve: cfg.AnswerReserve,
	})
	trace = append(trace, fmt.Sprintf(
		"budget: window=%d − system/instructions − query×2 − answer=%d ⇒ %d tok for evidence",
		cfg.ContextWindow, cfg.AnswerReserve, budget))

	// --- Step 1: cheap culling before anything expensive --------------------
	cur := Dedupe(hits, cfg.DedupeThreshold)
	trace = append(trace, fmt.Sprintf("dedupe: %d → %d chunks", len(hits), len(cur)))
	cur = CapPerSource(cur, cfg.MaxPerSource)
	trace = append(trace, fmt.Sprintf("source cap (≤%d/source): %d chunks, %d tok",
		cfg.MaxPerSource, len(cur), TotalTokens(cur, count)))

	// --- Step 2: compress only if over budget -------------------------------
	before := map[string]string{}
	for _, c := range cur {
		before[c.ID] = c.Text
	}
	cascade := CompressionCascade(cur, query, budget, count, summarize, cfg.ProtectTop)
	for _, s := range cascade.Stages {
		trace = append(trace, "  compress: "+s)
	}
	// Trimmed-but-verbatim vs paraphrased: the prompt must tell them apart, or
	// the model will "quote" a summary it invented the wording for.
	excerptIDs := map[string]bool{}
	for _, c := range cascade.Chunks {
		if before[c.ID] != c.Text && !cascade.Summarised[c.ID] {
			excerptIDs[c.ID] = true
		}
	}

	// --- Step 3: pack (compression may still leave us over) -----------------
	packed := PackGreedy(cascade.Chunks, budget, count)
	trace = append(trace, fmt.Sprintf("pack: %d/%d chunks kept, %d/%d tok used",
		len(packed), len(cascade.Chunks), TotalTokens(packed, count), budget))

	// --- Step 4: order for attention ----------------------------------------
	var ordered []Chunk
	switch cfg.Ordering {
	case "bookend":
		ordered = OrderBookend(packed)
	case "document":
		ordered = OrderByDocument(packed)
	case "grouped-bookend":
		ordered = OrderGroupedBookend(packed)
	default:
		ordered = OrderByRelevance(packed)
	}
	ids := make([]string, 0, len(ordered))
	for _, c := range ordered {
		ids = append(ids, c.ID)
	}
	trace = append(trace, fmt.Sprintf("order (%s): %s", cfg.Ordering, strings.Join(ids, " → ")))

	// --- Step 5: format with citation handles -------------------------------
	formatted := FormatForCitation(ordered, count, FormatOptions{
		Style:      cfg.Style,
		SummaryIDs: cascade.Summarised,
		ExcerptIDs: excerptIDs,
	})
	trace = append(trace, fmt.Sprintf("format: %d citable sources, %d tok with markup",
		len(formatted.Citations), formatted.Tokens))

	// --- Step 6: assemble, query-adjacent -----------------------------------
	layout := QueryAdjacentLayout(query, formatted.Text, instructions)
	prompt := fmt.Sprintf("%s\n\n%s\n\n%s", layout.Pre, layout.Context, layout.Post)
	trace = append(trace, fmt.Sprintf("prompt: %d tok total (answer reserve %d still free)",
		count(prompt), cfg.AnswerReserve))

	return BuildResult{Prompt: prompt, Context: formatted, Chunks: ordered, Trace: trace}
}
