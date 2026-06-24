// rewriters.go — all query-rewriting strategies for RAG.
//
// WHY REWRITE THE QUERY?
// ----------------------
// Users write short, conversational, or ambiguous questions.  The vector store
// holds long, formal, information-dense passages.  The vocabulary mismatch
// means the cosine distance between the question embedding and the right answer
// is often larger than the distance to a wrong but stylistically similar passage.
//
// Query rewriting bridges that gap BEFORE retrieval.
//
// STRATEGIES IMPLEMENTED
// ----------------------
//  1. KeywordExpand          — synonym injection; no LLM; zero latency
//  2. HyDE                   — generate a hypothetical answer; embed that
//  3. MultiQuery             — N paraphrase variants; union of results
//  4. StepBack               — rephrase to a broader, abstract question
//  5. SubQueryDecompose      — split into atomic sub-questions
//  6. RAGFusion              — multi-query + Reciprocal Rank Fusion
//  7. ContextualCompress     — condense chat history into a standalone query
//
// DESIGN PATTERN
// --------------
// Every LLM-dependent function accepts an LLMFn callback:
//
//	type LLMFn func(ctx context.Context, prompt string) (string, error)
//
// Pass in your real provider (Anthropic, OpenAI, Ollama) or a mock for tests.
package main

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

// LLMFn calls a language model with prompt and returns the text response.
// Implement this with your preferred provider.
type LLMFn func(ctx context.Context, prompt string) (string, error)

// RetrieverFn fetches ranked documents for query.
// Returns a slice of (docID, score) pairs, highest score first.
type RetrieverFn func(ctx context.Context, query string) ([]DocScore, error)

// DocScore pairs a document identifier with its retrieval score.
type DocScore struct {
	DocID string
	Score float64
}

// RewrittenQuery is the output of any rewriting strategy.
type RewrittenQuery struct {
	Query    string            // the rewritten (or expanded) query string
	Strategy string            // name of the strategy that produced it
	Metadata map[string]string // optional key-value pairs (e.g. "role": "abstract")
}

// ChatTurn is one message in a multi-turn conversation.
type ChatTurn struct {
	Role    string // "user" or "assistant"
	Content string
}

// FusedResult is one entry in a RAG-Fusion output.
type FusedResult struct {
	DocID     string
	RRFScore  float64
	AppearsIn int // how many query variants returned this document
}

// ---------------------------------------------------------------------------
// HELPER: split LLM output into one-item-per-line slices
// ---------------------------------------------------------------------------

// splitLines cleans up an LLM response where each logical item is on its own
// line.  It strips leading list markers (numbers, bullets, dashes) and
// surrounding whitespace, then filters empty lines.
func splitLines(text string) []string {
	var out []string
	for _, raw := range strings.Split(text, "\n") {
		// Remove common list prefixes: "1.", "1)", "-", "*", "•"
		line := strings.TrimSpace(raw)
		for _, prefix := range []string{"- ", "* ", "• "} {
			line = strings.TrimPrefix(line, prefix)
		}
		// Strip leading "1. " style numbering.
		if len(line) > 2 && line[1] == '.' || (len(line) > 3 && line[2] == '.') {
			if line[0] >= '0' && line[0] <= '9' {
				line = strings.TrimSpace(line[strings.Index(line, ".")+1:])
			}
		}
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

// =============================================================================
//  1. KEYWORD EXPANSION  (rule-based — no LLM)
// =============================================================================
//
//  WHAT IT IS
//      Inject synonyms for recognised words in the query so the embedding
//      covers a wider area of the semantic space.
//
//  HOW IT WORKS
//      For every token in the query that appears in a synonym map, append the
//      synonym list next to the original word.  Result: "improve|enhance|boost".
//
//  WHEN TO USE
//      - You have a narrow domain (legal, medical, finance) with stable terms.
//      - Zero latency is required — no network round-trip.
//
//  DOWNSIDE
//      The synonym map must be maintained by hand.  It misses novel or
//      context-dependent meanings.

// defaultSynonyms is a small built-in map for common RAG-domain terms.
var defaultSynonyms = map[string][]string{
	"fast":     {"quick", "rapid", "performant", "speedy"},
	"slow":     {"sluggish", "latent", "low-throughput"},
	"error":    {"bug", "exception", "failure", "fault", "issue"},
	"improve":  {"enhance", "optimize", "boost", "increase"},
	"database": {"db", "datastore", "storage", "repository"},
	"retrieve": {"fetch", "get", "query", "search", "find"},
	"document": {"doc", "file", "record", "text", "passage"},
}

// KeywordExpand expands query by injecting synonyms for known keywords.
// Extra entries in extraSynonyms override or extend the built-in map.
func KeywordExpand(query string, extraSynonyms map[string][]string) RewrittenQuery {
	// Merge built-in and caller-supplied synonym maps.
	merged := make(map[string][]string, len(defaultSynonyms)+len(extraSynonyms))
	for k, v := range defaultSynonyms {
		merged[k] = v
	}
	for k, v := range extraSynonyms {
		merged[k] = v
	}

	words := strings.Fields(query)
	expanded := make([]string, 0, len(words))

	for _, word := range words {
		// Normalise to lowercase, strip punctuation, for the map lookup.
		key := strings.ToLower(strings.Trim(word, ".,!?;:\"'"))
		if syns, ok := merged[key]; ok {
			// "word|syn1|syn2" format — keep original + alternatives.
			expanded = append(expanded, strings.Join(append([]string{word}, syns...), "|"))
		} else {
			expanded = append(expanded, word)
		}
	}

	return RewrittenQuery{
		Query:    strings.Join(expanded, " "),
		Strategy: "keyword-expansion",
		Metadata: map[string]string{"originalQuery": query},
	}
}

// =============================================================================
//  2. HyDE — Hypothetical Document Embeddings
// =============================================================================
//
//  WHAT IT IS
//      Ask the LLM for a short hypothetical answer, then embed THAT instead of
//      the bare question.  The fake answer lives in the same vector space as
//      real documents, so nearest-neighbour search retrieves documents similar
//      to "what the answer looks like" rather than "what the question looks like".
//
//  WHEN TO USE
//      - Short, keyword-heavy queries that stylistically differ from your corpus.
//      - Dense retrieval over long-form documents (articles, manuals, papers).
//
//  REFERENCE  Gao et al., 2022 "Precise Zero-Shot Dense Retrieval without
//             Relevance Labels" (arxiv 2212.10496)

// HyDE generates a hypothetical answer passage for query and returns it as the
// RewrittenQuery.Query. The caller should embed this string, not the original query.
func HyDE(ctx context.Context, query string, llm LLMFn, maxWords int) (RewrittenQuery, error) {
	if maxWords <= 0 {
		maxWords = 120
	}
	prompt := "Write a factual passage of at most " + itoa(maxWords) + " words that directly " +
		"answers the following question. Do NOT include the question itself.\n\n" +
		"Question: " + query + "\n\nPassage:"

	hypothetical, err := llm(ctx, prompt)
	if err != nil {
		return RewrittenQuery{}, err
	}

	return RewrittenQuery{
		// The hypothetical passage IS the query — embed this, not the original.
		Query:    strings.TrimSpace(hypothetical),
		Strategy: "hyde",
		Metadata: map[string]string{"originalQuery": query},
	}, nil
}

// =============================================================================
//  3. MULTI-QUERY (LLM-based paraphrase expansion)
// =============================================================================
//
//  WHAT IT IS
//      Ask the LLM to rephrase the question N different ways.  Run retrieval
//      for EACH variant, then union the result sets.
//
//  WHEN TO USE
//      - Underspecified or multi-interpretable queries.
//      - You want to increase recall cheaply (before applying a re-ranker).
//
//  DOWNSIDE
//      N × retrieval latency.  Naïve union can dilute precision.
//      Use RAGFusion (strategy 6) to fix the dilution problem.

// MultiQuery generates numVariants paraphrases of query.
// The original query is always included as index 0.
func MultiQuery(ctx context.Context, query string, llm LLMFn, numVariants int) ([]RewrittenQuery, error) {
	if numVariants <= 0 {
		numVariants = 3
	}
	prompt := "Generate " + itoa(numVariants) + " different phrasings of the following question. " +
		"Each should preserve the original intent but use different words or structure. " +
		"Output ONLY the questions, one per line, no numbering or bullets.\n\n" +
		"Original question: " + query

	raw, err := llm(ctx, prompt)
	if err != nil {
		return nil, err
	}

	variants := splitLines(raw)
	if len(variants) > numVariants {
		variants = variants[:numVariants]
	}

	// Start with the original so it is always represented.
	results := []RewrittenQuery{
		{
			Query:    query,
			Strategy: "multi-query",
			Metadata: map[string]string{"variant": "0", "isOriginal": "true"},
		},
	}
	for i, v := range variants {
		results = append(results, RewrittenQuery{
			Query:    v,
			Strategy: "multi-query",
			Metadata: map[string]string{"variant": itoa(i + 1), "isOriginal": "false"},
		})
	}
	return results, nil
}

// =============================================================================
//  4. STEP-BACK PROMPTING
// =============================================================================
//
//  WHAT IT IS
//      Rephrase the specific question into a more ABSTRACT question that
//      retrieves background / foundational knowledge.  Use BOTH queries.
//
//  EXAMPLE
//      Original : "What was the temperature in Miami on 1 Jan 2022?"
//      Step-back: "What factors determine temperatures in subtropical cities?"
//
//  REFERENCE  Zheng et al., 2023 "Take a Step Back: Evoking Reasoning in LLMs
//             via Abstraction" (arxiv 2310.06117)

// StepBack returns two queries: the original (role="specific") and a broader
// abstract rewrite (role="abstract").  Retrieve documents for both and merge.
func StepBack(ctx context.Context, query string, llm LLMFn) ([]RewrittenQuery, error) {
	prompt := "Given the following specific question, write ONE more general question " +
		"that captures the underlying concept or background knowledge required to answer it. " +
		"Output ONLY the general question, nothing else.\n\n" +
		"Specific question: " + query + "\n\nGeneral question:"

	stepBack, err := llm(ctx, prompt)
	if err != nil {
		return nil, err
	}

	return []RewrittenQuery{
		{Query: query, Strategy: "step-back", Metadata: map[string]string{"role": "specific"}},
		{Query: strings.TrimSpace(stepBack), Strategy: "step-back", Metadata: map[string]string{"role": "abstract"}},
	}, nil
}

// =============================================================================
//  5. SUB-QUERY DECOMPOSITION
// =============================================================================
//
//  WHAT IT IS
//      Break a complex, multi-faceted question into SIMPLER atomic sub-
//      questions.  Retrieve documents per sub-question, then synthesise.
//
//  EXAMPLE
//      Original: "Compare PostgreSQL vs MongoDB for e-commerce."
//      Sub-queries:
//        1. "PostgreSQL performance characteristics for reads"
//        2. "MongoDB performance characteristics for reads"
//        3. "e-commerce read workload patterns"
//
//  WHEN TO USE
//      - Comparative, multi-part, or "explain everything about X and Y" questions.

// SubQueryDecompose breaks query into at most maxSubQueries atomic sub-questions.
func SubQueryDecompose(ctx context.Context, query string, llm LLMFn, maxSubQueries int) ([]RewrittenQuery, error) {
	if maxSubQueries <= 0 {
		maxSubQueries = 5
	}
	prompt := "Break the following complex question into at most " + itoa(maxSubQueries) +
		" simpler, self-contained sub-questions. Each must be independently answerable. " +
		"Output ONLY the sub-questions, one per line, no numbering or bullets.\n\n" +
		"Complex question: " + query

	raw, err := llm(ctx, prompt)
	if err != nil {
		return nil, err
	}

	subQueries := splitLines(raw)
	if len(subQueries) > maxSubQueries {
		subQueries = subQueries[:maxSubQueries]
	}

	results := make([]RewrittenQuery, len(subQueries))
	for i, q := range subQueries {
		results[i] = RewrittenQuery{
			Query:    q,
			Strategy: "sub-query-decomposition",
			Metadata: map[string]string{
				"subQueryIndex":    itoa(i),
				"totalSubQueries":  itoa(len(subQueries)),
			},
		}
	}
	return results, nil
}

// =============================================================================
//  6. RAG-FUSION  (Multi-Query + Reciprocal Rank Fusion)
// =============================================================================
//
//  WHAT IT IS
//      Multi-Query (strategy 3) + RRF to merge N ranked result lists into one
//      re-ranked list.  RRF prevents naïve union from over-promoting documents
//      that appear in many weak result sets.
//
//  RRF FORMULA  (Cormack et al., SIGIR 2009)
//      RRF_score(d) = Σ  1 / (k + rank_i(d))
//      where rank_i(d) is the 1-based rank of d in list i (∞ if absent),
//      and k = 60 is the standard smoothing constant.
//
//  REFERENCE  Shi et al., 2023 "RAG-Fusion: Improving LLM-based Retrieval
//             Augmented Generation Systems"

// RAGFusion generates numQueries variants, retrieves via retriever for each,
// and fuses the ranked lists with Reciprocal Rank Fusion.
// k is the RRF smoothing constant (default 60 if zero).
func RAGFusion(
	ctx context.Context,
	query string,
	llm LLMFn,
	retriever RetrieverFn,
	numQueries int,
	k float64,
) ([]FusedResult, error) {
	if numQueries <= 0 {
		numQueries = 4
	}
	if k <= 0 {
		k = 60
	}

	// Step 1 — Generate query variants (numQueries-1 paraphrases + original).
	variants, err := MultiQuery(ctx, query, llm, numQueries-1)
	if err != nil {
		return nil, err
	}
	queries := make([]string, len(variants))
	for i, v := range variants {
		queries[i] = v.Query
	}

	// Step 2 — Retrieve ranked lists for each variant in parallel would be
	// ideal; for clarity here we do it sequentially.
	rrfScores := map[string]float64{}
	appearsIn := map[string]int{}

	for _, q := range queries {
		ranked, err := retriever(ctx, q)
		if err != nil {
			return nil, err
		}
		for rank, ds := range ranked {
			// RRF uses 1-based ranking.
			contribution := 1.0 / (k + float64(rank+1))
			rrfScores[ds.DocID] += contribution
			appearsIn[ds.DocID]++
		}
	}

	// Step 3 — Collect into a slice and sort descending by RRF score.
	fused := make([]FusedResult, 0, len(rrfScores))
	for docID, score := range rrfScores {
		fused = append(fused, FusedResult{
			DocID:     docID,
			RRFScore:  math.Round(score*1e6) / 1e6, // round to 6 decimals for readability
			AppearsIn: appearsIn[docID],
		})
	}
	sort.Slice(fused, func(i, j int) bool {
		return fused[i].RRFScore > fused[j].RRFScore
	})
	return fused, nil
}

// =============================================================================
//  7. CONTEXTUAL COMPRESSION (chat history → standalone query)
// =============================================================================
//
//  WHAT IT IS
//      In multi-turn chat, the latest message may use pronouns or references
//      that depend on earlier turns ("What about its scalability?" — "its"
//      refers to PostgreSQL from three turns ago).  The retriever has no
//      memory of prior turns, so the pronoun-heavy question returns garbage.
//
//      Contextual compression rewrites the latest message into a STANDALONE
//      question that a fresh retriever can understand with no context.
//
//  WHEN TO USE
//      ALWAYS in conversational / chat-over-documents interfaces.
//      The LLM cost is negligible; the retrieval precision gain is large.

// ContextualCompress rewrites the last user turn in history into a standalone
// retrieval query.  history must contain at least one user turn.
func ContextualCompress(ctx context.Context, history []ChatTurn, llm LLMFn) (RewrittenQuery, error) {
	// Find the last user message — that is the question being rewritten.
	var lastUser string
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			lastUser = history[i].Content
			break
		}
	}
	if lastUser == "" {
		return RewrittenQuery{}, fmt.Errorf("contextualCompress: no user turn found in history")
	}

	// Build a readable conversation transcript.
	var sb strings.Builder
	for _, t := range history {
		role := "User"
		if t.Role == "assistant" {
			role = "Assistant"
		}
		sb.WriteString(role + ": " + t.Content + "\n")
	}

	prompt := "Given the following conversation, rewrite the final user question as a " +
		"STANDALONE question that contains all context needed for a search engine — " +
		"no pronouns like \"it\", \"they\", \"the above\". " +
		"Output ONLY the rewritten question.\n\n" +
		"Conversation:\n" + sb.String() + "\nStandalone question:"

	rewritten, err := llm(ctx, prompt)
	if err != nil {
		return RewrittenQuery{}, err
	}

	return RewrittenQuery{
		Query:    strings.TrimSpace(rewritten),
		Strategy: "contextual-compression",
		Metadata: map[string]string{
			"originalQuestion": lastUser,
			"historyLength":    itoa(len(history)),
		},
	}, nil
}

// ---------------------------------------------------------------------------
// tiny helper — avoids importing strconv just for Itoa
// ---------------------------------------------------------------------------
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
