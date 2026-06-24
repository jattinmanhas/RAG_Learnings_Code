// ============================================================================
//  CHUNKING STRATEGIES  (Go)
// ============================================================================
//
//  WHY CHUNK AT ALL?
//  -----------------
//  In a RAG pipeline, once you have raw text (see "3.1 Document Preprocessing"),
//  you cannot embed a whole 50-page document as one vector. Two reasons:
//
//    1. EMBEDDING MODELS HAVE A TOKEN LIMIT. Text beyond the limit is silently
//       truncated — you lose information.
//    2. RETRIEVAL PRECISION. One vector for a huge document averages everything
//       together, so a search for a niche fact returns a blurry match. Smaller,
//       focused chunks give sharper, more relevant retrieval.
//
//  So we split text into "chunks": pieces small enough to embed, large enough
//  to still carry meaning. HOW we split is the art — that's this file.
//
//  THE CORE TENSION (memorize this)
//  --------------------------------
//      Too SMALL  -> each chunk lacks context, meaning gets fragmented.
//      Too LARGE  -> retrieval is imprecise, you waste the LLM's context window.
//      No OVERLAP -> a fact split across a boundary is lost to both chunks.
//
//  Every strategy below is a different answer to "where do I cut, and how do I
//  avoid cutting through the middle of an idea?"
//
//  NOTE ON UNICODE: we operate on []rune (not bytes) so multi-byte characters
//  are never sliced in half.
// ============================================================================

package main

import (
	"math"
	"regexp"
	"strings"
)

// ----------------------------------------------------------------------------
// A SHARED SHAPE
// ----------------------------------------------------------------------------
// Every chunker returns the SAME type, so downstream code (embedding, storage)
// doesn't care which strategy produced the chunks. Metadata is where we stash
// useful breadcrumbs (e.g. which heading a chunk came from).
type Chunk struct {
	Text     string                 // the actual text of this chunk
	Index    int                    // position of this chunk in the document (0-based)
	Start    int                    // rune offset where this chunk starts
	End      int                    // rune offset where this chunk ends
	Metadata map[string]interface{} // optional extra info — section title, token count…
}

// makeChunk trims a slice and only keeps it if it has real content.
func makeChunk(text string, index, start, end int, metadata map[string]interface{}) (Chunk, bool) {
	trimmed := strings.TrimSpace(text)
	if len(trimmed) == 0 {
		return Chunk{}, false
	}
	return Chunk{Text: trimmed, Index: index, Start: start, End: end, Metadata: metadata}, true
}

/* ===========================================================================
 *  1. FIXED-SIZE CHUNKING (by characters)
 * ===========================================================================
 *  Cut the text every N runes. Blind, simple, fast.
 *
 *  HOW IT WORKS
 *      "abcdefghij", size 4  ->  ["abcd", "efgh", "ij"]
 *
 *  WHEN TO USE
 *      - Quick prototypes / baselines.
 *      - Text with NO structure (logs, OCR dumps, transcripts).
 *
 *  DOWNSIDE
 *      It happily slices through the middle of words and sentences. That's what
 *      OVERLAP (#2) and the smarter splitters (#5) are designed to fix.
 * =========================================================================== */
func FixedSizeChunk(text string, size int) []Chunk {
	runes := []rune(text)
	var chunks []Chunk
	index := 0
	for start := 0; start < len(runes); start += size {
		end := min(start+size, len(runes))
		if c, ok := makeChunk(string(runes[start:end]), index, start, end, nil); ok {
			chunks = append(chunks, c)
			index++
		}
	}
	return chunks
}

/* ===========================================================================
 *  2. FIXED-SIZE WITH OVERLAP (sliding window)
 * ===========================================================================
 *  Same as #1, but each chunk repeats the last `overlap` runes of the previous
 *  one. The window slides forward by (size - overlap).
 *
 *  HOW IT WORKS  (size 6, overlap 2)
 *      "abcdefghij"  ->  ["abcdef", "efghij", "ij"]
 *
 *  WHY OVERLAP MATTERS
 *      A sentence (or a name, or a number) that lands on a boundary survives in
 *      AT LEAST ONE chunk intact, instead of being cut in half and lost to both.
 *      Rule of thumb: overlap = 10–20% of chunk size.
 *
 *  WHEN TO USE
 *      - The sensible DEFAULT for fixed-size approaches.
 *      - Any time context bleeds across boundaries (most prose).
 * =========================================================================== */
func FixedSizeChunkWithOverlap(text string, size, overlap int) []Chunk {
	if overlap >= size {
		panic("overlap must be smaller than size")
	}
	runes := []rune(text)
	var chunks []Chunk
	step := size - overlap // how far the window advances each time
	index := 0
	for start := 0; start < len(runes); start += step {
		end := min(start+size, len(runes))
		if c, ok := makeChunk(string(runes[start:end]), index, start, end, nil); ok {
			chunks = append(chunks, c)
			index++
		}
		if end == len(runes) {
			break // avoid emitting tiny trailing duplicates
		}
	}
	return chunks
}

var sentenceRe = regexp.MustCompile(`[^.!?]+[.!?]+|\S+$`)
var whitespaceRe = regexp.MustCompile(`\s+`)

/* ===========================================================================
 *  3. SENTENCE-BASED CHUNKING
 * ===========================================================================
 *  Split into sentences first, then GROUP sentences together until we hit a
 *  target size. We never cut mid-sentence — boundaries land on "." "?" "!".
 *
 *  WHEN TO USE
 *      - Well-punctuated prose (articles, docs, books).
 *      - When you want chunks that read like complete thoughts.
 *
 *  DOWNSIDE
 *      The naive regex treats "Dr." or "3.14" as sentence ends. Real systems use
 *      an NLP sentence tokenizer for abbreviation-aware splitting. Teaching version.
 * =========================================================================== */
func SentenceChunk(text string, maxChars int) []Chunk {
	flat := whitespaceRe.ReplaceAllString(text, " ")
	sentences := sentenceRe.FindAllString(flat, -1)

	var chunks []Chunk
	var buffer strings.Builder
	index, cursor := 0, 0

	flush := func() {
		body := buffer.String()
		start := cursor
		end := cursor + len([]rune(body))
		if c, ok := makeChunk(body, index, start, end, nil); ok {
			chunks = append(chunks, c)
			index++
		}
		cursor = end
		buffer.Reset()
	}

	for _, s := range sentences {
		s = strings.TrimSpace(s)
		if buffer.Len() > 0 && buffer.Len()+len(s) > maxChars {
			flush()
		}
		if buffer.Len() > 0 {
			buffer.WriteString(" ")
		}
		buffer.WriteString(s)
	}
	if buffer.Len() > 0 {
		flush()
	}
	return chunks
}

var blankLineRe = regexp.MustCompile(`\n\s*\n`)

/* ===========================================================================
 *  4. PARAGRAPH-BASED CHUNKING
 * ===========================================================================
 *  Authors already grouped related ideas into paragraphs — respect that. Split
 *  on blank lines, then pack paragraphs together up to a size budget.
 *
 *  WHEN TO USE
 *      - Structured prose where blank lines mark topic shifts (Markdown,
 *        well-formatted docs, blog posts).
 *
 *  DOWNSIDE
 *      A single giant paragraph can blow past the budget. We fall back to
 *      splitting such monsters with overlap so nothing is left oversized.
 * =========================================================================== */
func ParagraphChunk(text string, maxChars int) []Chunk {
	var paragraphs []string
	for _, p := range blankLineRe.Split(text, -1) {
		if t := strings.TrimSpace(p); t != "" {
			paragraphs = append(paragraphs, t)
		}
	}

	var chunks []Chunk
	var buffer strings.Builder
	index, cursor := 0, 0

	flush := func() {
		body := buffer.String()
		start := cursor
		end := cursor + len([]rune(body))
		if c, ok := makeChunk(body, index, start, end, nil); ok {
			chunks = append(chunks, c)
			index++
		}
		cursor = end
		buffer.Reset()
	}

	for _, para := range paragraphs {
		// A paragraph that alone exceeds the budget: split it, flush what we have.
		if len(para) > maxChars {
			if buffer.Len() > 0 {
				flush()
			}
			for _, sub := range FixedSizeChunkWithOverlap(para, maxChars, int(float64(maxChars)*0.1)) {
				start := cursor
				end := cursor + len([]rune(sub.Text))
				chunks = append(chunks, Chunk{Text: sub.Text, Index: index, Start: start, End: end})
				index++
				cursor = end
			}
			continue
		}
		if buffer.Len() > 0 && buffer.Len()+len(para) > maxChars {
			flush()
		}
		if buffer.Len() > 0 {
			buffer.WriteString("\n\n")
		}
		buffer.WriteString(para)
	}
	if buffer.Len() > 0 {
		flush()
	}
	return chunks
}

/* ===========================================================================
 *  5. RECURSIVE CHARACTER SPLITTING  ⭐ (the workhorse)
 * ===========================================================================
 *  The strategy LangChain popularized as RecursiveCharacterTextSplitter, and
 *  the best general-purpose default.
 *
 *  THE IDEA
 *      Try to split on the BIGGEST natural boundary first (paragraphs). If a
 *      piece is still too big, recurse and split it on the next boundary
 *      (lines, sentences), then words, then — last resort — raw characters.
 *
 *      separators = ["\n\n", "\n", ". ", " ", ""]
 *
 *  WHY IT'S GOOD
 *      It keeps semantically-related text together as much as the size budget
 *      allows, only making "ugly" cuts when there's no cleaner option.
 *
 *  WHEN TO USE
 *      - Your DEFAULT for mixed / unknown content. When in doubt, use this.
 * =========================================================================== */
func RecursiveChunk(text string, maxChars, overlap int, separators []string) []Chunk {
	if separators == nil {
		separators = []string{"\n\n", "\n", ". ", " ", ""}
	}

	rawPieces := recursiveSplit(text, maxChars, separators)

	// Re-attach overlap (carry the tail of the previous piece) + offsets.
	var chunks []Chunk
	cursor, index := 0, 0
	prevTail := ""
	for _, piece := range rawPieces {
		withOverlap := piece
		if prevTail != "" {
			withOverlap = prevTail + " " + piece
		}
		start := cursor
		end := cursor + len([]rune(withOverlap))
		if c, ok := makeChunk(withOverlap, index, start, end, nil); ok {
			chunks = append(chunks, c)
			index++
		}
		cursor += len([]rune(piece))
		if overlap > 0 {
			pr := []rune(piece)
			if len(pr) > overlap {
				prevTail = string(pr[len(pr)-overlap:])
			} else {
				prevTail = piece
			}
		} else {
			prevTail = ""
		}
	}
	return chunks
}

func recursiveSplit(input string, maxChars int, seps []string) []string {
	if len([]rune(input)) <= maxChars {
		return []string{input}
	}
	sep := seps[0]
	rest := seps[1:]

	var pieces []string
	if sep == "" {
		pieces = hardSplit(input, maxChars) // give up gracefully: cut by runes
	} else {
		pieces = strings.Split(input, sep)
	}

	var out []string
	buffer := ""
	for _, piece := range pieces {
		candidate := piece
		if buffer != "" {
			candidate = buffer + sep + piece
		}
		if len([]rune(candidate)) <= maxChars {
			buffer = candidate
		} else {
			if buffer != "" {
				out = append(out, buffer)
			}
			if len([]rune(piece)) > maxChars {
				next := rest
				if len(next) == 0 {
					next = []string{""}
				}
				out = append(out, recursiveSplit(piece, maxChars, next)...)
				buffer = ""
			} else {
				buffer = piece
			}
		}
	}
	if buffer != "" {
		out = append(out, buffer)
	}
	return out
}

func hardSplit(input string, size int) []string {
	runes := []rune(input)
	var out []string
	for i := 0; i < len(runes); i += size {
		out = append(out, string(runes[i:min(i+size, len(runes))]))
	}
	return out
}

/* ===========================================================================
 *  6. TOKEN-BASED CHUNKING
 * ===========================================================================
 *  Embedding models count TOKENS, not characters. If your limit is in tokens,
 *  chunk in tokens to never overflow the model.
 *
 *  REAL SYSTEMS use the model's actual tokenizer (tiktoken for OpenAI, the
 *  HuggingFace tokenizer otherwise). Here we APPROXIMATE: ~1 token ≈ 4 chars of
 *  English so the demo has zero dependencies. Swap EstimateTokens for a real
 *  tokenizer in production.
 *
 *  WHEN TO USE
 *      - You must respect a hard token limit (embedding model / LLM context).
 *      - Cost control: you're billed per token, so token-sized chunks are
 *        predictable.
 * =========================================================================== */
const charsPerToken = 4 // crude English approximation

func EstimateTokens(text string) int {
	return int(math.Ceil(float64(len([]rune(text))) / charsPerToken))
}

func TokenChunk(text string, maxTokens, overlapTokens int) []Chunk {
	// Convert the token budget into an approximate character budget and reuse
	// the overlap logic. In production, slice on real token boundaries instead.
	sizeChars := maxTokens * charsPerToken
	overlapChars := overlapTokens * charsPerToken
	chunks := FixedSizeChunkWithOverlap(text, sizeChars, overlapChars)
	for i := range chunks {
		chunks[i].Metadata = map[string]interface{}{"approxTokens": EstimateTokens(chunks[i].Text)}
	}
	return chunks
}

var headingRe = regexp.MustCompile(`^(#{1,6})\s+(.*)$`)

/* ===========================================================================
 *  7. STRUCTURE-AWARE / MARKDOWN-HEADER CHUNKING
 * ===========================================================================
 *  Documents with headings carry their own outline. Splitting on headings
 *  produces chunks that map 1:1 to sections, and we stash the heading TRAIL
 *  ("Guide > Setup > Install") into metadata — gold for retrieval and for
 *  showing users where an answer came from.
 *
 *  WHEN TO USE
 *      - Markdown, HTML, wikis, API docs — anything with a heading hierarchy.
 *      - When you want chunk metadata that preserves document structure.
 *
 *  DOWNSIDE
 *      A section under one heading can still be huge -> we sub-split big
 *      sections with the recursive splitter (#5) so size stays bounded.
 * =========================================================================== */
type heading struct {
	level int
	title string
}

func MarkdownHeaderChunk(text string, maxChars int) []Chunk {
	lines := strings.Split(text, "\n")
	var chunks []Chunk
	index, cursor := 0, 0
	var stack []heading
	var sectionLines []string

	flushSection := func() {
		body := strings.TrimSpace(strings.Join(sectionLines, "\n"))
		sectionLines = nil
		if body == "" {
			return
		}
		titles := make([]string, len(stack))
		for i, h := range stack {
			titles[i] = h.title
		}
		trail := strings.Join(titles, " > ")
		if trail == "" {
			trail = "(root)"
		}

		var pieces []Chunk
		if len([]rune(body)) > maxChars {
			pieces = RecursiveChunk(body, maxChars, 50, nil)
		} else {
			pieces = []Chunk{{Text: body}}
		}
		for _, piece := range pieces {
			start := cursor
			end := cursor + len([]rune(piece.Text))
			chunks = append(chunks, Chunk{
				Text: piece.Text, Index: index, Start: start, End: end,
				Metadata: map[string]interface{}{"headingTrail": trail},
			})
			index++
			cursor = end
		}
	}

	for _, line := range lines {
		if m := headingRe.FindStringSubmatch(line); m != nil {
			flushSection() // close out the section we were accumulating
			level := len(m[1])
			title := strings.TrimSpace(m[2])
			// Pop deeper-or-equal headings, then push this one (keeps the trail).
			for len(stack) > 0 && stack[len(stack)-1].level >= level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, heading{level: level, title: title})
		} else {
			sectionLines = append(sectionLines, line)
		}
	}
	flushSection()
	return chunks
}

/* ===========================================================================
 *  8. SEMANTIC CHUNKING
 * ===========================================================================
 *  The most "intelligent" approach: split where the MEANING shifts, not where
 *  the character count happens to land.
 *
 *  HOW IT WORKS
 *      1. Break text into sentences.
 *      2. Embed each sentence (turn it into a vector).
 *      3. Walk sentence-by-sentence, measuring similarity between consecutive
 *         sentences.
 *      4. When similarity DROPS below a threshold -> topic boundary -> new chunk.
 *
 *  WHEN TO USE
 *      - High-value corpora where retrieval quality justifies the extra cost.
 *      - Content that wanders between topics without clear structure
 *        (interview transcripts, meeting notes, long essays).
 *
 *  COST / DOWNSIDE
 *      You pay to embed every sentence BEFORE you even chunk. Slower and pricier
 *      than the others. Often overkill — try recursive (#5) first.
 *
 *  NOTE: takes an EmbedFn callback so this file stays dependency-free. In a real
 *  app you'd pass your embedding model's function (OpenAI / Cohere / local).
 * =========================================================================== */
type EmbedFn func(texts []string) ([][]float64, error)

func cosineSimilarity(a, b []float64) float64 {
	var dot, magA, magB float64
	for i := range a {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	denom := math.Sqrt(magA) * math.Sqrt(magB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func SemanticChunk(text string, embed EmbedFn, similarityThreshold float64, maxChars int) ([]Chunk, error) {
	flat := whitespaceRe.ReplaceAllString(text, " ")
	var sentences []string
	for _, s := range sentenceRe.FindAllString(flat, -1) {
		if t := strings.TrimSpace(s); t != "" {
			sentences = append(sentences, t)
		}
	}
	if len(sentences) == 0 {
		return nil, nil
	}

	vectors, err := embed(sentences)
	if err != nil {
		return nil, err
	}

	var chunks []Chunk
	buffer := sentences[0]
	index, cursor := 0, 0

	flush := func() {
		start := cursor
		end := cursor + len([]rune(buffer))
		if c, ok := makeChunk(buffer, index, start, end, nil); ok {
			chunks = append(chunks, c)
			index++
		}
		cursor = end
	}

	for i := 1; i < len(sentences); i++ {
		sim := cosineSimilarity(vectors[i-1], vectors[i])
		wouldOverflow := len(buffer)+len(sentences[i]) > maxChars
		// New chunk when the topic shifts (low similarity) OR we'd overflow.
		if sim < similarityThreshold || wouldOverflow {
			flush()
			buffer = sentences[i]
		} else {
			buffer += " " + sentences[i]
		}
	}
	flush()
	return chunks, nil
}
