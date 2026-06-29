// retriever.go — Small-to-Big (Sentence-Window) Retrieval for RAG
//
// THE CORE PROBLEM
// ----------------
// The same chunking dilemma every RAG system faces:
//
//   Small chunks (~1 sentence):
//     ✓ Tight, focused embeddings — great retrieval precision
//     ✗ Too little surrounding context for the LLM to reason from
//
//   Large chunks (~a paragraph or more):
//     ✓ Full context — the LLM can reason coherently
//     ✗ Embedding is a blurry average of many ideas — retrieval suffers
//
// SMALL-TO-BIG RESOLVES THIS — AND DIFFERS FROM PARENT-CHILD (3.4.1)
// -----------------------------------------------------------------
// Parent-child retrieval (3.4.1) splits each document into a FIXED set of
// parent sections at ingest time, sub-splits parents into children, and at
// query time maps every matched child back to its predefined parent.
//
// Small-to-big retrieval takes a different, more granular tack:
//
//   • The atomic unit is a single SENTENCE (the "small" chunk).
//   • Every sentence is embedded and indexed, tagged with its position
//     (docID + ordinal) in the original document.
//   • There are NO predefined parents.  At query time, for each matched
//     sentence we DYNAMICALLY expand a window of ±N neighbouring sentences
//     around it — the "big" context — by walking the original sentence list.
//   • Overlapping windows are MERGED so the LLM never sees a sentence twice.
//
// In short:
//   Parent-child  → static boundaries, map child → fixed parent.
//   Small-to-big  → dynamic boundaries, expand sentence → sliding window,
//                   then merge overlaps. The "big" chunk is centred exactly
//                   on the hit, not snapped to a section boundary.
//
// ARCHITECTURE
// ------------
//
// INGESTION TIME:
//   Document  (an ordered list of sentences)
//      │
//      ├── S0  ← embedded + indexed (pos 0)
//      ├── S1  ← embedded + indexed (pos 1)
//      ├── S2  ← embedded + indexed (pos 2)
//      ├── S3  ← embedded + indexed (pos 3)
//      └── ...                       (the FULL ordered list is also kept,
//                                     so we can expand windows at query time)
//
// QUERY TIME (windowSize = 1):
//   Query → embed → vector search (sentence index)
//      → S5 scores 0.91
//      → S6 scores 0.88
//      → S12 scores 0.81
//              │
//              ▼
//   Expand each hit to a window of ±1 sentence:
//      → S5  → window [S4, S5, S6]
//      → S6  → window [S5, S6, S7]
//      → S12 → window [S11, S12, S13]
//              │
//              ▼
//   Merge overlapping windows (S5's and S6's windows touch → [S4..S7]):
//      → Window A: S4 S5 S6 S7   (from hits S5, S6)
//      → Window B: S11 S12 S13   (from hit S12)
//              │
//              ▼
//   Feed the merged windows to the LLM  ← contiguous context, no fragments,
//                                          no repeated sentences.
//
// DESIGN DECISIONS
// ----------------
//  • The sentence index is the ONLY thing queried at search time.
//  • Window expansion is bounded to the SAME document (no cross-doc bleed).
//  • Merging is interval-union: any two windows that touch or overlap become
//    one contiguous block, scored by the best hit inside it.
//  • EmbedFn is an injectable callback — swap in OpenAI, Cohere, Voyage, or a
//    mock without touching retrieval logic.
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

// Sentence is the atomic "small" unit — what actually gets embedded and
// searched.  It remembers where it lives so we can expand a window later.
type Sentence struct {
	ID    string // e.g. "doc1-sent-7"
	DocID string // which document this sentence belongs to
	Pos   int    // 0-based ordinal of this sentence within its document
	Text  string // the sentence text (the thing that gets embedded)
}

// SentenceScore pairs a sentence with its cosine-similarity score from the
// vector search.  Sorted descending — highest score = best match.
type SentenceScore struct {
	Sentence Sentence
	Score    float64
}

// WindowResult is the "big" context handed to the LLM: a contiguous block of
// sentences expanded around one or more matched sentences, with overlapping
// windows already merged.
type WindowResult struct {
	DocID          string   // document this window came from
	StartPos       int      // position of the first sentence in the window
	EndPos         int      // position of the last sentence in the window
	Text           string   // the joined sentence text — what the LLM reads
	BestScore      float64  // best hit score that contributed to this window
	MatchedSentIDs []string // IDs of the sentence hits inside this window
}

// EmbedFn converts any text string into a float64 vector.
// Implement this with your real embedding provider (OpenAI, Cohere, Voyage, …).
// In tests and demos, pass a mock that returns deterministic vectors.
type EmbedFn func(ctx context.Context, text string) ([]float64, error)

// ---------------------------------------------------------------------------
// IN-MEMORY STORE
// ---------------------------------------------------------------------------

// SentenceStore holds the two data structures needed at query time:
//   - sentences: the flat "vector index" of every sentence across all docs.
//                In production this is a pgvector table with an embedding
//                column and an HNSW index.
//   - byDoc:     docID → the document's sentences in original order.  This is
//                what lets us expand a ±N window around any hit.  In
//                production this is a plain Postgres table keyed by
//                (doc_id, position) — no vector column needed.
type SentenceStore struct {
	sentences []Sentence
	byDoc     map[string][]Sentence
}

// NewStore creates an empty SentenceStore.
func NewStore() *SentenceStore {
	return &SentenceStore{
		sentences: nil,
		byDoc:     make(map[string][]Sentence),
	}
}

// ---------------------------------------------------------------------------
// INGESTION
// ---------------------------------------------------------------------------

// IngestDocument splits a document into sentences, assigns each a position,
// and stores them both in the flat index (for search) and grouped by document
// in original order (for window expansion).
//
// After IngestDocument, call embed() on every sentence and upsert those
// vectors into your vector store before running any queries.
func IngestDocument(store *SentenceStore, docID string, text string) {
	sentTexts := splitIntoSentences(text)
	docSents := make([]Sentence, 0, len(sentTexts))
	for i, s := range sentTexts {
		sent := Sentence{
			ID:    fmt.Sprintf("%s-sent-%d", docID, i),
			DocID: docID,
			Pos:   i,
			Text:  s,
		}
		store.sentences = append(store.sentences, sent)
		docSents = append(docSents, sent)
	}
	store.byDoc[docID] = docSents
}

// splitIntoSentences breaks text into trimmed, non-empty sentences by walking
// the runes and ending a sentence at . ! ? when followed by whitespace.
//
// This is the minimal viable splitter. Go's regexp (RE2) has no lookbehind,
// so we tokenize manually. In production use a real sentence tokenizer (e.g.
// an ICU segmenter) that handles abbreviations, decimals, and quotes.
func splitIntoSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	var sentences []string
	var b strings.Builder
	runes := []rune(text)
	for i, r := range runes {
		b.WriteRune(r)
		if r == '.' || r == '!' || r == '?' {
			// Peek: a sentence ends here only if the next rune is whitespace
			// or we're at the end of the text.
			if i+1 >= len(runes) || isSpace(runes[i+1]) {
				s := strings.TrimSpace(b.String())
				if s != "" {
					sentences = append(sentences, s)
				}
				b.Reset()
			}
		}
	}
	// Trailing text with no terminal punctuation (e.g. a heading line).
	if tail := strings.TrimSpace(b.String()); tail != "" {
		sentences = append(sentences, tail)
	}
	return sentences
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

// ---------------------------------------------------------------------------
// VECTOR SEARCH (sentence index)
// ---------------------------------------------------------------------------

// searchSentences performs a similarity search over all sentences and returns
// the topK best matches, sorted descending by score.
//
// In production this is a single SQL query:
//
//	SELECT id, doc_id, pos, text,
//	       1 - (embedding <=> $queryVec) AS score
//	FROM   sentences
//	ORDER  BY embedding <=> $queryVec
//	LIMIT  $topK;
//
// Here we compute cosine similarity in Go using the mock embedding function.
func searchSentences(
	ctx context.Context,
	query string,
	store *SentenceStore,
	embed EmbedFn,
	topK int,
) ([]SentenceScore, error) {
	if topK <= 0 {
		topK = 6
	}

	queryVec, err := embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	scored := make([]SentenceScore, 0, len(store.sentences))
	for _, sent := range store.sentences {
		sentVec, err := embed(ctx, sent.Text)
		if err != nil {
			return nil, fmt.Errorf("embed sentence %s: %w", sent.ID, err)
		}
		scored = append(scored, SentenceScore{
			Sentence: sent,
			Score:    cosineSimilarity(queryVec, sentVec),
		})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	if len(scored) > topK {
		scored = scored[:topK]
	}
	return scored, nil
}

// cosineSimilarity returns the cosine similarity between two equal-length
// vectors a and b (range [−1, 1]; higher = more similar).
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ---------------------------------------------------------------------------
// WINDOW EXPANSION + MERGE  (the "small → big" step)
// ---------------------------------------------------------------------------

// span is an internal half-open interval [start, end] of sentence positions
// within a single document, carrying the best contributing hit score and the
// IDs of the hits that fall inside it.
type span struct {
	docID     string
	start     int
	end       int
	bestScore float64
	hitIDs    []string
}

// expandAndMerge is the heart of small-to-big retrieval.
//
//  1. For each sentence hit, expand to a window [pos-windowSize, pos+windowSize],
//     clamped to the document's bounds.
//  2. Group the windows by document and sort by start position.
//  3. Merge any windows that overlap OR are adjacent (touching) into a single
//     contiguous span — this is an interval-union. The merged span's score is
//     the best score among the hits it absorbed.
//  4. Materialise each span into a WindowResult by joining the actual sentence
//     text from store.byDoc.
//
// Results are returned sorted by BestScore descending, so the strongest match
// leads the context fed to the LLM.
func expandAndMerge(
	store *SentenceStore,
	hits []SentenceScore,
	windowSize int,
) []WindowResult {
	if windowSize < 0 {
		windowSize = 1
	}

	// Step 1: expand each hit into a clamped window span.
	spansByDoc := make(map[string][]span)
	for _, h := range hits {
		docSents, ok := store.byDoc[h.Sentence.DocID]
		if !ok {
			continue // orphaned hit — data integrity issue; skip gracefully
		}
		start := h.Sentence.Pos - windowSize
		if start < 0 {
			start = 0
		}
		end := h.Sentence.Pos + windowSize
		if end > len(docSents)-1 {
			end = len(docSents) - 1
		}
		spansByDoc[h.Sentence.DocID] = append(spansByDoc[h.Sentence.DocID], span{
			docID:     h.Sentence.DocID,
			start:     start,
			end:       end,
			bestScore: h.Score,
			hitIDs:    []string{h.Sentence.ID},
		})
	}

	// Steps 2 + 3: within each document, sort by start and merge overlaps.
	var merged []span
	for docID, spans := range spansByDoc {
		sort.Slice(spans, func(i, j int) bool {
			return spans[i].start < spans[j].start
		})
		cur := spans[0]
		for _, s := range spans[1:] {
			// Touching or overlapping → union them. "+1" so windows that are
			// merely adjacent (e.g. [4,6] and [7,9]) still merge into one block.
			if s.start <= cur.end+1 {
				if s.end > cur.end {
					cur.end = s.end
				}
				if s.bestScore > cur.bestScore {
					cur.bestScore = s.bestScore
				}
				cur.hitIDs = append(cur.hitIDs, s.hitIDs...)
			} else {
				merged = append(merged, cur)
				cur = s
			}
		}
		merged = append(merged, cur)
		_ = docID
	}

	// Step 4: materialise spans into WindowResults with the joined text.
	results := make([]WindowResult, 0, len(merged))
	for _, s := range merged {
		docSents := store.byDoc[s.docID]
		texts := make([]string, 0, s.end-s.start+1)
		for p := s.start; p <= s.end; p++ {
			texts = append(texts, docSents[p].Text)
		}
		results = append(results, WindowResult{
			DocID:          s.docID,
			StartPos:       s.start,
			EndPos:         s.end,
			Text:           strings.Join(texts, " "),
			BestScore:      s.bestScore,
			MatchedSentIDs: s.hitIDs,
		})
	}

	// Order by best score descending — strongest context first.
	sort.Slice(results, func(i, j int) bool {
		return results[i].BestScore > results[j].BestScore
	})
	return results
}

// ---------------------------------------------------------------------------
// PUBLIC RETRIEVAL PIPELINE
// ---------------------------------------------------------------------------

// Retrieve is the complete small-to-big retrieval pipeline:
//
//  1. Embed the query using embed().
//  2. Search the sentence vector index for the topK most similar sentences.
//  3. Expand each matched sentence to a ±windowSize window of neighbours.
//  4. Merge overlapping/adjacent windows into contiguous blocks.
//  5. Return the merged windows ordered by best sentence similarity score.
//
// The caller passes each window's Text directly to the LLM as context.
//
// Parameters:
//   - topKSentences: how many sentences to retrieve before window expansion.
//   - windowSize:    number of neighbouring sentences to include on EACH side
//                    of a hit (1 → 3-sentence windows, 2 → 5-sentence, …).
//   - maxWindows:    upper bound on returned windows (0 = no limit).
func Retrieve(
	ctx context.Context,
	store *SentenceStore,
	query string,
	embed EmbedFn,
	topKSentences int,
	windowSize int,
	maxWindows int,
) ([]WindowResult, error) {
	// Step 1 + 2: embed query → search sentence index.
	hits, err := searchSentences(ctx, query, store, embed, topKSentences)
	if err != nil {
		return nil, fmt.Errorf("sentence search: %w", err)
	}

	// Step 3 + 4: expand each hit to a window, then merge overlaps.
	results := expandAndMerge(store, hits, windowSize)

	// Step 5: apply maxWindows cap.
	if maxWindows > 0 && len(results) > maxWindows {
		results = results[:maxWindows]
	}
	return results, nil
}
