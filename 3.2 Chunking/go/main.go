// ============================================================================
//  CHUNKING DEMO  —  run with:  go run .
// ============================================================================
//
//  This runs every strategy from chunkers.go on the same sample text so you can
//  SEE how they differ. Read the output side-by-side with the comments in
//  chunkers.go.
//
//  CHEAT SHEET — which strategy should I reach for?
//  ------------------------------------------------
//    Fixed-size .............. quick baseline; unstructured text (logs, OCR).
//    Fixed + overlap ......... default for fixed approaches; stops boundary loss.
//    Sentence ................ well-punctuated prose; want whole thoughts.
//    Paragraph ............... blank-line-structured docs; respect author intent.
//    Recursive ⭐ ............ BEST general default for mixed/unknown content.
//    Token ................... must respect a hard token limit / control cost.
//    Markdown-header ......... docs/wikis with headings; want section metadata.
//    Semantic ................ high-value corpora; topic-drifting text; costlier.
// ============================================================================

package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

const sample = `# Retrieval-Augmented Generation

RAG combines a retriever with a generator. The retriever finds relevant text. The generator writes an answer grounded in that text.

## Why Chunking Matters

Embedding models have token limits. You cannot embed a whole book at once. Smaller chunks also make retrieval sharper, because each vector represents one focused idea instead of a blurry average.

## Trade-offs

Chunks that are too small lose context. Chunks that are too large waste the context window and dilute relevance. Overlap helps facts survive across boundaries.`

func preview(text string, n int) string {
	flat := whitespaceRe.ReplaceAllString(text, " ")
	r := []rune(flat)
	if len(r) > n {
		return string(r[:n]) + "…"
	}
	return flat
}

func report(title string, chunks []Chunk) {
	bar := strings.Repeat("=", 76)
	fmt.Printf("\n%s\n%s\n%s\n", bar, title, bar)
	fmt.Printf("  %d chunk(s):\n", len(chunks))
	for _, c := range chunks {
		meta := ""
		if c.Metadata != nil {
			b, _ := json.Marshal(c.Metadata)
			meta = "  " + string(b)
		}
		fmt.Printf("  [%d] (%d chars) %q%s\n", c.Index, len([]rune(c.Text)), preview(c.Text, 60), meta)
	}
}

// fakeEmbed is a dependency-free stand-in so the semantic demo runs with no API
// key. It maps text to a tiny keyword-presence vector — enough to make the
// topic-boundary logic visible. Replace with a real embedder in production.
func fakeEmbed(texts []string) ([][]float64, error) {
	topics := []string{"retriev", "embed", "chunk", "context", "overlap", "answer"}
	out := make([][]float64, len(texts))
	for i, t := range texts {
		lower := strings.ToLower(t)
		vec := make([]float64, len(topics))
		for j, kw := range topics {
			if strings.Contains(lower, kw) {
				vec[j] = 1
			}
		}
		out[i] = vec
	}
	return out, nil
}

func main() {
	report("1. FIXED-SIZE (size=200)", FixedSizeChunk(sample, 200))
	report("2. FIXED-SIZE + OVERLAP (size=200, overlap=40)", FixedSizeChunkWithOverlap(sample, 200, 40))
	report("3. SENTENCE (maxChars=200)", SentenceChunk(sample, 200))
	report("4. PARAGRAPH (maxChars=300)", ParagraphChunk(sample, 300))
	report("5. RECURSIVE ⭐ (maxChars=200, overlap=30)", RecursiveChunk(sample, 200, 30, nil))
	report("6. TOKEN (maxTokens=60, overlap=10)", TokenChunk(sample, 60, 10))
	report("7. MARKDOWN-HEADER (maxChars=400)", MarkdownHeaderChunk(sample, 400))

	semantic, err := SemanticChunk(sample, fakeEmbed, 0.75, 400)
	if err != nil {
		panic(err)
	}
	report("8. SEMANTIC (threshold=0.75)", semantic)
}
