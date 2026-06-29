// retriever.go — Parent-Child Retrieval for RAG
//
// THE CORE PROBLEM
// ----------------
// Every RAG system faces a chunking dilemma:
//
//   Small chunks (~150 tokens):
//     ✓ Tight, focused embeddings — great retrieval precision
//     ✗ Too little surrounding context for the LLM to reason from
//
//   Large chunks (~800 tokens):
//     ✓ Full section context — the LLM can reason coherently
//     ✗ Embedding is a blurry average of many sub-topics — retrieval suffers
//
// PARENT-CHILD RETRIEVAL RESOLVES THIS BY SEPARATING THE CONCERN
// ---------------------------------------------------------------
//   SEARCH  with child embeddings  → precise vector match
//   FEED    parent chunks to LLM  → full reasoning context
//
// ARCHITECTURE
// ------------
//
// INGESTION TIME:
//   Document
//      │
//      ├── Parent 1 (full section, ~800 tokens)  ← stored in a map, NOT embedded
//      │       ├── Child 1.1 (~150 tokens)  ← embedded + indexed in vector store
//      │       ├── Child 1.2 (~150 tokens)  ← embedded + indexed in vector store
//      │       └── Child 1.3 (~150 tokens)  ← embedded + indexed in vector store
//      │
//      ├── Parent 2 (full section, ~700 tokens)  ← stored, NOT embedded
//      │       ├── Child 2.1 (~150 tokens)  ← embedded + indexed
//      │       ├── Child 2.2 (~150 tokens)  ← embedded + indexed
//      │       └── Child 2.3 (~150 tokens)  ← embedded + indexed
//      └── ...
//
// QUERY TIME:
//   Query → embed → vector search (child index only)
//      → Child 2.1 scores 0.91
//      → Child 2.2 scores 0.87
//      → Child 1.3 scores 0.82
//              │
//              ▼
//   Fetch parent of each match
//      → Parent 2  (from Child 2.1 and 2.2)
//      → Parent 1  (from Child 1.3)
//              │
//              ▼
//   Deduplicate by parentID  ← key step: 2.1 and 2.2 both point to Parent 2,
//                                         returned only ONCE
//              │
//              ▼
//   Feed Parent 1 + Parent 2 to LLM  ← full context, no fragments
//
// DESIGN DECISIONS
// ----------------
//  • The child index is the ONLY thing queried at search time.
//    Parents are never put in the vector store.
//  • Parent deduplication is stable: if N children from the same parent all
//    match, the parent appears once, ordered by the best child score.
//  • EmbedFn is an injectable callback — swap in OpenAI, Cohere, or a mock
//    without touching retrieval logic.
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

// ParentChunk is a large section of a document — what the LLM ultimately reads.
// It is stored in memory (or a key-value store) keyed by its ID.
// It is NEVER embedded or added to the vector index.
type ParentChunk struct {
	ID    string // e.g. "doc1-parent-0"
	Title string // section heading, if available
	Text  string // full section text (~700-900 tokens in production)
}

// ChildChunk is a short passage derived from a parent.
// ONLY children are embedded and indexed in the vector store.
// Each child carries a ParentID so we can retrieve the parent later.
type ChildChunk struct {
	ID       string // e.g. "doc1-parent-0-child-1"
	ParentID string // foreign key back to the parent
	Text     string // the short passage that gets embedded (~100-200 tokens)
}

// ChildScore is a child chunk paired with the cosine-similarity score from
// the vector search.  Sorted descending — highest score = best match.
type ChildScore struct {
	Child ChildChunk
	Score float64
}

// ParentResult is what the retrieval pipeline hands to the LLM.
// It carries the full parent text plus diagnostic info about which children
// triggered the match and how confident those matches were.
type ParentResult struct {
	Parent          ParentChunk
	BestChildScore  float64  // the highest similarity score among matched children
	MatchedChildren []string // IDs of child chunks that pointed to this parent
}

// EmbedFn converts any text string into a float64 vector.
// Implement this with your real embedding provider (OpenAI, Cohere, Voyage, …).
// In tests and demos, pass a mock that returns deterministic vectors.
type EmbedFn func(ctx context.Context, text string) ([]float64, error)

// ---------------------------------------------------------------------------
// IN-MEMORY STORE
// ---------------------------------------------------------------------------

// ParentChildStore holds the two data structures needed at query time:
//   - parents: a map from parentID → full ParentChunk text  (key-value lookup)
//   - children: the "vector index" — in production this lives in pgvector,
//               Pinecone, Weaviate, etc.  Here we store the raw text and
//               compute similarity on the fly in the mock.
//
// In a production system:
//   - parents  → Redis, DynamoDB, or a Postgres table (no vector column)
//   - children → pgvector table with an HNSW index on the embedding column
type ParentChildStore struct {
	parents  map[string]ParentChunk
	children []ChildChunk // in production: each row has an embedding column
}

// NewStore creates an empty ParentChildStore.
func NewStore() *ParentChildStore {
	return &ParentChildStore{
		parents:  make(map[string]ParentChunk),
		children: nil,
	}
}

// ---------------------------------------------------------------------------
// INGESTION
// ---------------------------------------------------------------------------

// IngestDocument splits a document into parent chunks and child chunks,
// then stores parents in the map and appends children to the vector index.
//
// Parameters:
//   - docID:       unique document identifier (used to namespace chunk IDs)
//   - parentTexts: pre-split section texts that become parent chunks.
//                  In production, you'd use a semantic splitter from 3.2.
//   - childSize:   approximate token budget per child chunk (used to sub-split
//                  each parent into smaller passages).
//
// After IngestDocument, call embed() on every child and upsert those vectors
// into your vector store before running any queries.
func IngestDocument(
	store *ParentChildStore,
	docID string,
	parentTexts []string,
	childSize int, // approximate words per child (tokens ≈ words * 1.3)
) {
	for i, pText := range parentTexts {
		parentID := fmt.Sprintf("%s-parent-%d", docID, i)

		// Store the full parent text — this is what the LLM will read.
		store.parents[parentID] = ParentChunk{
			ID:    parentID,
			Title: extractTitle(pText),
			Text:  pText,
		}

		// Split the parent into smaller child passages for embedding.
		childTexts := splitIntoChildren(pText, childSize)
		for j, cText := range childTexts {
			store.children = append(store.children, ChildChunk{
				ID:       fmt.Sprintf("%s-child-%d", parentID, j),
				ParentID: parentID,
				Text:     cText,
			})
		}
	}
}

// splitIntoChildren naively splits text into roughly equal passages of ≤ maxWords
// words each.  In a real system you'd use the sentence-aware or semantic splitter
// from chapter 3.2 (overlap-aware, respects sentence boundaries).
func splitIntoChildren(text string, maxWords int) []string {
	if maxWords <= 0 {
		maxWords = 100
	}
	words := strings.Fields(text)
	var chunks []string
	for i := 0; i < len(words); i += maxWords {
		end := i + maxWords
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}
	return chunks
}

// extractTitle attempts to pull a section heading from the first line of text.
func extractTitle(text string) string {
	lines := strings.SplitN(strings.TrimSpace(text), "\n", 2)
	if len(lines) == 0 {
		return ""
	}
	first := strings.TrimSpace(lines[0])
	// Treat lines shorter than 80 chars that end without a period as titles.
	if len(first) < 80 && !strings.HasSuffix(first, ".") {
		return first
	}
	return ""
}

// ---------------------------------------------------------------------------
// VECTOR SEARCH (child index)
// ---------------------------------------------------------------------------

// searchChildren performs a similarity search over all child chunks and
// returns the topK best matches, sorted descending by score.
//
// In production this is a single SQL query:
//
//	SELECT id, parent_id, text,
//	       1 - (embedding <=> $queryVec) AS score
//	FROM   child_chunks
//	ORDER  BY embedding <=> $queryVec
//	LIMIT  $topK;
//
// Here we compute cosine similarity in Go using the mock embedding function.
func searchChildren(
	ctx context.Context,
	query string,
	store *ParentChildStore,
	embed EmbedFn,
	topK int,
) ([]ChildScore, error) {
	if topK <= 0 {
		topK = 6
	}

	// Embed the query.
	queryVec, err := embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// Score every child by cosine similarity.
	scored := make([]ChildScore, 0, len(store.children))
	for _, child := range store.children {
		childVec, err := embed(ctx, child.Text)
		if err != nil {
			return nil, fmt.Errorf("embed child %s: %w", child.ID, err)
		}
		score := cosineSimilarity(queryVec, childVec)
		scored = append(scored, ChildScore{Child: child, Score: score})
	}

	// Sort descending by score.
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
// PARENT FETCH + DEDUPLICATION
// ---------------------------------------------------------------------------

// fetchAndDedup takes the top-k child results, looks up each child's parent,
// and deduplicates: if multiple children point to the same parent, that parent
// appears only ONCE in the output, annotated with all matching child IDs and
// ordered by the best (highest) child score.
//
// This is the key step that distinguishes parent-child retrieval from naïve
// large-chunk retrieval: we searched precisely (children) but return context
// generously (parents).
func fetchAndDedup(
	store *ParentChildStore,
	childScores []ChildScore,
) []ParentResult {
	// Track which parents we've seen to avoid duplicates.
	seen := make(map[string]*ParentResult)
	var order []string // preserves insertion order (best-score-first)

	for _, cs := range childScores {
		parentID := cs.Child.ParentID
		parent, ok := store.parents[parentID]
		if !ok {
			// Orphaned child — data integrity issue; skip gracefully.
			continue
		}

		if existing, found := seen[parentID]; found {
			// Parent already in the result set.  Just add the child ID to
			// the matched list; the BestChildScore was already set when the
			// parent first appeared (since childScores is sorted descending,
			// the first encounter is always the best score).
			existing.MatchedChildren = append(existing.MatchedChildren, cs.Child.ID)
		} else {
			result := &ParentResult{
				Parent:          parent,
				BestChildScore:  cs.Score,
				MatchedChildren: []string{cs.Child.ID},
			}
			seen[parentID] = result
			order = append(order, parentID)
		}
	}

	// Reconstruct the slice in insertion order (= best-score-first, because
	// childScores was already sorted descending before we iterated).
	results := make([]ParentResult, 0, len(order))
	for _, pid := range order {
		results = append(results, *seen[pid])
	}
	return results
}

// ---------------------------------------------------------------------------
// PUBLIC RETRIEVAL PIPELINE
// ---------------------------------------------------------------------------

// Retrieve is the complete parent-child retrieval pipeline:
//
//  1. Embed the query using embed().
//  2. Search the child vector index for the topK most similar child chunks.
//  3. Look up the parent of every matched child.
//  4. Deduplicate parents (multiple children → same parent → one entry).
//  5. Return parent chunks ordered by best child similarity score.
//
// The caller passes these parent texts directly to the LLM as context.
//
// Parameters:
//   - topKChildren: how many children to retrieve before parent-mapping.
//                   More children = higher recall, more parent chunks returned.
//   - maxParents:   upper bound on returned parent chunks (0 = no limit).
func Retrieve(
	ctx context.Context,
	store *ParentChildStore,
	query string,
	embed EmbedFn,
	topKChildren int,
	maxParents int,
) ([]ParentResult, error) {
	// Step 1 + 2: embed query → search child index.
	childScores, err := searchChildren(ctx, query, store, embed, topKChildren)
	if err != nil {
		return nil, fmt.Errorf("child search: %w", err)
	}

	// Step 3 + 4: map children → parents, deduplicate.
	results := fetchAndDedup(store, childScores)

	// Step 5: apply maxParents cap.
	if maxParents > 0 && len(results) > maxParents {
		results = results[:maxParents]
	}
	return results, nil
}
