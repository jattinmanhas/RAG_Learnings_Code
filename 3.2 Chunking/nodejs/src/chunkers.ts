/**
 * ============================================================================
 *  CHUNKING STRATEGIES  (Node.js / TypeScript)
 * ============================================================================
 *
 *  WHY CHUNK AT ALL?
 *  -----------------
 *  In a RAG pipeline, once you have raw text (see "3.1 Document Preprocessing"),
 *  you cannot embed a whole 50-page document as one vector. Two reasons:
 *
 *    1. EMBEDDING MODELS HAVE A TOKEN LIMIT. Text beyond the limit is silently
 *       truncated — you lose information.
 *    2. RETRIEVAL PRECISION. One vector for a huge document averages everything
 *       together, so a search for a niche fact returns a blurry match. Smaller,
 *       focused chunks give sharper, more relevant retrieval.
 *
 *  So we split text into "chunks": pieces small enough to embed, large enough
 *  to still carry meaning. HOW we split is the art — that's what this file is.
 *
 *  THE CORE TENSION (memorize this)
 *  --------------------------------
 *      Too SMALL  -> each chunk lacks context, meaning gets fragmented.
 *      Too LARGE  -> retrieval is imprecise, you waste the LLM's context window.
 *      No OVERLAP -> a fact split across a boundary is lost to both chunks.
 *
 *  Every strategy below is a different answer to "where do I cut, and how do I
 *  avoid cutting through the middle of an idea?"
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// A SHARED SHAPE
// ----------------------------------------------------------------------------
// Every chunker returns the SAME type, so downstream code (embedding, storage)
// doesn't care which strategy produced the chunks. `metadata` is where we stash
// useful breadcrumbs (e.g. which heading a chunk came from).
export interface Chunk {
  /** The actual text of this chunk. */
  text: string;
  /** Index of this chunk within the document (0-based). */
  index: number;
  /** Character offset where this chunk starts in the original text. */
  start: number;
  /** Character offset where this chunk ends in the original text. */
  end: number;
  /** Optional extra info — section title, token count, etc. */
  metadata?: Record<string, unknown>;
}

// Small helper: trim a slice and only keep it if it has real content.
function makeChunk(
  text: string,
  index: number,
  start: number,
  end: number,
  metadata?: Record<string, unknown>
): Chunk | null {
  const trimmed = text.trim();
  if (trimmed.length === 0) return null;
  return { text: trimmed, index, start, end, metadata };
}

/* ===========================================================================
 *  1. FIXED-SIZE CHUNKING (by characters)
 * ===========================================================================
 *  Cut the text every N characters. Blind, simple, fast.
 *
 *  HOW IT WORKS
 *      "abcdefghij", size 4  ->  ["abcd", "efgh", "ij"]
 *
 *  WHEN TO USE
 *      - Quick prototypes / baselines.
 *      - Text with NO structure (logs, OCR dumps, transcripts without
 *        punctuation).
 *
 *  DOWNSIDE
 *      It happily slices through the middle of words and sentences
 *      ("...the impor" | "tant thing..."), which can confuse the embedding.
 *      That's exactly what OVERLAP (strategy #2) and the smarter splitters
 *      (#5) are designed to fix.
 * =========================================================================== */
export function fixedSizeChunk(text: string, size = 500): Chunk[] {
  const chunks: Chunk[] = [];
  let index = 0;
  for (let start = 0; start < text.length; start += size) {
    const end = Math.min(start + size, text.length);
    const chunk = makeChunk(text.slice(start, end), index, start, end);
    if (chunk) chunks.push((index++, chunk));
  }
  return chunks;
}

/* ===========================================================================
 *  2. FIXED-SIZE WITH OVERLAP (sliding window)
 * ===========================================================================
 *  Same as #1, but each chunk repeats the last `overlap` characters of the
 *  previous one. The window slides forward by (size - overlap).
 *
 *  HOW IT WORKS  (size 6, overlap 2)
 *      "abcdefghij"  ->  ["abcdef", "efghij", "ij"]
 *                              ^^         ^^   shared tails/heads
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
export function fixedSizeChunkWithOverlap(
  text: string,
  size = 500,
  overlap = 75
): Chunk[] {
  if (overlap >= size) throw new Error("overlap must be smaller than size");
  const chunks: Chunk[] = [];
  const step = size - overlap; // how far the window advances each time
  let index = 0;
  for (let start = 0; start < text.length; start += step) {
    const end = Math.min(start + size, text.length);
    const chunk = makeChunk(text.slice(start, end), index, start, end);
    if (chunk) chunks.push((index++, chunk));
    if (end === text.length) break; // avoid emitting tiny trailing duplicates
  }
  return chunks;
}

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
 *      The naive regex below treats "Dr." or "3.14" as sentence ends. Real
 *      systems use an NLP sentence tokenizer (e.g. wink-nlp, spaCy) for
 *      abbreviation-aware splitting. This is the teaching version.
 * =========================================================================== */
export function sentenceChunk(text: string, maxChars = 500): Chunk[] {
  // Split AFTER end-punctuation followed by whitespace. The lookbehind keeps
  // the punctuation attached to its sentence.
  const sentences = text
    .replace(/\s+/g, " ")
    .match(/[^.!?]+[.!?]+|\S+$/g) ?? [];

  const chunks: Chunk[] = [];
  let buffer = "";
  let index = 0;
  let cursor = 0; // approximate char offset into original text

  const flush = () => {
    const start = cursor;
    const end = cursor + buffer.length;
    const chunk = makeChunk(buffer, index, start, end);
    if (chunk) {
      chunks.push(chunk);
      index++;
    }
    cursor = end;
    buffer = "";
  };

  for (const sentence of sentences) {
    // If adding this sentence would overflow, close the current chunk first.
    if (buffer.length + sentence.length > maxChars && buffer.length > 0) {
      flush();
    }
    buffer += (buffer ? " " : "") + sentence.trim();
  }
  if (buffer) flush();
  return chunks;
}

/* ===========================================================================
 *  4. PARAGRAPH-BASED CHUNKING
 * ===========================================================================
 *  Authors already grouped related ideas into paragraphs — respect that.
 *  Split on blank lines, then pack paragraphs together up to a size budget.
 *
 *  WHEN TO USE
 *      - Structured prose where blank lines mark topic shifts (Markdown,
 *        well-formatted docs, blog posts).
 *
 *  DOWNSIDE
 *      A single giant paragraph can blow past your size budget. We fall back to
 *      splitting such monsters with overlap so nothing is left oversized.
 * =========================================================================== */
export function paragraphChunk(text: string, maxChars = 800): Chunk[] {
  const paragraphs = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);

  const chunks: Chunk[] = [];
  let buffer = "";
  let index = 0;
  let cursor = 0;

  const flush = () => {
    const start = cursor;
    const end = cursor + buffer.length;
    const chunk = makeChunk(buffer, index, start, end);
    if (chunk) {
      chunks.push(chunk);
      index++;
    }
    cursor = end;
    buffer = "";
  };

  for (const para of paragraphs) {
    // A paragraph that alone exceeds the budget: split it, flush what we have.
    if (para.length > maxChars) {
      if (buffer) flush();
      for (const sub of fixedSizeChunkWithOverlap(para, maxChars, Math.floor(maxChars * 0.1))) {
        const start = cursor;
        const end = cursor + sub.text.length;
        chunks.push({ ...sub, index: index++, start, end });
        cursor = end;
      }
      continue;
    }
    if (buffer.length + para.length > maxChars && buffer.length > 0) flush();
    buffer += (buffer ? "\n\n" : "") + para;
  }
  if (buffer) flush();
  return chunks;
}

/* ===========================================================================
 *  5. RECURSIVE CHARACTER SPLITTING  ⭐ (the workhorse)
 * ===========================================================================
 *  This is the strategy LangChain popularized as RecursiveCharacterTextSplitter,
 *  and it's the best general-purpose default.
 *
 *  THE IDEA
 *      Try to split on the BIGGEST natural boundary first (paragraphs). If a
 *      piece is still too big, recurse and split it on the next boundary
 *      (sentences), then words, then — as a last resort — raw characters.
 *
 *      separators = ["\n\n", "\n", ". ", " ", ""]
 *                      |      |     |     |    |
 *                  paragraph line  sentence word  character (give up gracefully)
 *
 *  WHY IT'S GOOD
 *      It keeps semantically-related text together as much as the size budget
 *      allows, only making "ugly" cuts when there's no cleaner option. It
 *      adapts to whatever structure the text actually has.
 *
 *  WHEN TO USE
 *      - Your DEFAULT for mixed / unknown content. When in doubt, use this.
 * =========================================================================== */
export function recursiveChunk(
  text: string,
  maxChars = 500,
  overlap = 50,
  separators: string[] = ["\n\n", "\n", ". ", " ", ""]
): Chunk[] {
  // Internal recursion returns plain strings; we attach offsets/indices after.
  function split(input: string, seps: string[]): string[] {
    if (input.length <= maxChars) return [input];

    const [sep, ...rest] = seps;
    // "" means we've exhausted separators -> hard-cut by characters.
    const pieces = sep === "" ? hardSplit(input, maxChars) : input.split(sep);

    const out: string[] = [];
    let buffer = "";
    for (const piece of pieces) {
      const candidate = buffer ? buffer + sep + piece : piece;
      if (candidate.length <= maxChars) {
        buffer = candidate;
      } else {
        if (buffer) out.push(buffer);
        // The piece itself may still be too big -> recurse with finer separators.
        if (piece.length > maxChars) {
          out.push(...split(piece, rest.length ? rest : [""]));
          buffer = "";
        } else {
          buffer = piece;
        }
      }
    }
    if (buffer) out.push(buffer);
    return out;
  }

  function hardSplit(input: string, size: number): string[] {
    const out: string[] = [];
    for (let i = 0; i < input.length; i += size) out.push(input.slice(i, i + size));
    return out;
  }

  const rawPieces = split(text, separators);

  // Re-attach overlap (carry the tail of the previous piece) + offsets.
  const chunks: Chunk[] = [];
  let cursor = 0;
  let index = 0;
  let prevTail = "";
  for (const piece of rawPieces) {
    const withOverlap = prevTail ? prevTail + " " + piece : piece;
    const start = cursor;
    const end = cursor + withOverlap.length;
    const chunk = makeChunk(withOverlap, index, start, end);
    if (chunk) {
      chunks.push(chunk);
      index++;
    }
    cursor += piece.length;
    prevTail = overlap > 0 ? piece.slice(-overlap) : "";
  }
  return chunks;
}

/* ===========================================================================
 *  6. TOKEN-BASED CHUNKING
 * ===========================================================================
 *  Embedding models count TOKENS, not characters. "chunking" might be 1 token;
 *  "antidisestablishmentarianism" might be 6. If your limit is in tokens, you
 *  should chunk in tokens to never overflow the model.
 *
 *  REAL SYSTEMS use the model's actual tokenizer:
 *      - OpenAI:  `tiktoken` / `js-tiktoken`
 *      - Others:  the HuggingFace tokenizer for that model
 *  Here we APPROXIMATE: ~1 token ≈ 4 characters of English (a well-known
 *  rule of thumb) so the demo has zero dependencies. Swap `estimateTokens`
 *  for a real tokenizer in production.
 *
 *  WHEN TO USE
 *      - You must respect a hard token limit (embedding model, or LLM context).
 *      - Cost control: you're billed per token, so token-sized chunks are
 *        predictable.
 * =========================================================================== */
const CHARS_PER_TOKEN = 4; // crude English approximation

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function tokenChunk(text: string, maxTokens = 256, overlapTokens = 25): Chunk[] {
  // Convert the token budget into an approximate character budget and reuse
  // the overlap logic. In production, slice on real token boundaries instead.
  const sizeChars = maxTokens * CHARS_PER_TOKEN;
  const overlapChars = overlapTokens * CHARS_PER_TOKEN;
  return fixedSizeChunkWithOverlap(text, sizeChars, overlapChars).map((c) => ({
    ...c,
    metadata: { ...c.metadata, approxTokens: estimateTokens(c.text) },
  }));
}

/* ===========================================================================
 *  7. STRUCTURE-AWARE / MARKDOWN-HEADER CHUNKING
 * ===========================================================================
 *  Documents with headings carry their own outline. Splitting on headings
 *  produces chunks that map 1:1 to sections, and we can stash the heading
 *  TRAIL ("Guide > Setup > Install") into metadata — gold for retrieval, and
 *  great for showing users "where" an answer came from.
 *
 *  WHEN TO USE
 *      - Markdown, HTML, wikis, API docs, anything with a heading hierarchy.
 *      - When you want chunk metadata that preserves document structure.
 *
 *  DOWNSIDE
 *      A section under one heading can still be huge -> we sub-split big
 *      sections with the recursive splitter (#5) so size stays bounded.
 * =========================================================================== */
export function markdownHeaderChunk(text: string, maxChars = 800): Chunk[] {
  const lines = text.split("\n");
  const chunks: Chunk[] = [];
  let index = 0;
  let cursor = 0;

  // Track the current heading at each level to build a breadcrumb trail.
  const headingStack: { level: number; title: string }[] = [];
  let sectionLines: string[] = [];

  const flushSection = () => {
    const body = sectionLines.join("\n").trim();
    sectionLines = [];
    if (!body) return;

    const trail = headingStack.map((h) => h.title).join(" > ");

    // Section too big? Recursively split it, but keep the heading trail.
    const pieces = body.length > maxChars ? recursiveChunk(body, maxChars, 50) : [
      { text: body, index: 0, start: 0, end: body.length } as Chunk,
    ];

    for (const piece of pieces) {
      const start = cursor;
      const end = cursor + piece.text.length;
      chunks.push({
        text: piece.text,
        index: index++,
        start,
        end,
        metadata: { headingTrail: trail || "(root)" },
      });
      cursor = end;
    }
  };

  for (const line of lines) {
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      // Close out whatever section we were accumulating.
      flushSection();
      const level = heading[1].length;
      const title = heading[2].trim();
      // Pop deeper-or-equal headings, then push this one (maintains the trail).
      while (headingStack.length && headingStack[headingStack.length - 1].level >= level) {
        headingStack.pop();
      }
      headingStack.push({ level, title });
    } else {
      sectionLines.push(line);
    }
  }
  flushSection();
  return chunks;
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
 *      3. Walk sentence-by-sentence, measuring the similarity between
 *         consecutive sentences.
 *      4. When similarity DROPS below a threshold, that's a topic boundary —
 *         start a new chunk there.
 *
 *  WHEN TO USE
 *      - High-value corpora where retrieval quality justifies the extra cost.
 *      - Content that wanders between topics without clear structure
 *        (interview transcripts, meeting notes, long essays).
 *
 *  COST / DOWNSIDE
 *      You pay to embed every sentence BEFORE you even chunk. Slower and more
 *      expensive than the others. Often overkill — try recursive (#5) first.
 *
 *  NOTE: the function below takes an `embed` callback so this file stays
 *  dependency-free. In a real app you'd pass your embedding model's function
 *  (e.g. one that calls the OpenAI / Cohere / local-model embeddings API).
 * =========================================================================== */
export type EmbedFn = (texts: string[]) => Promise<number[][]>;

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) || 1);
}

export async function semanticChunk(
  text: string,
  embed: EmbedFn,
  options: { similarityThreshold?: number; maxChars?: number } = {}
): Promise<Chunk[]> {
  const { similarityThreshold = 0.75, maxChars = 1000 } = options;

  const sentences = (text.replace(/\s+/g, " ").match(/[^.!?]+[.!?]+|\S+$/g) ?? [])
    .map((s) => s.trim())
    .filter(Boolean);
  if (sentences.length === 0) return [];

  const vectors = await embed(sentences);

  const chunks: Chunk[] = [];
  let buffer = sentences[0];
  let index = 0;
  let cursor = 0;

  const flush = () => {
    const start = cursor;
    const end = cursor + buffer.length;
    const chunk = makeChunk(buffer, index, start, end);
    if (chunk) {
      chunks.push(chunk);
      index++;
    }
    cursor = end;
  };

  for (let i = 1; i < sentences.length; i++) {
    const sim = cosineSimilarity(vectors[i - 1], vectors[i]);
    const wouldOverflow = buffer.length + sentences[i].length > maxChars;
    // New chunk when the topic shifts (low similarity) OR we'd overflow.
    if (sim < similarityThreshold || wouldOverflow) {
      flush();
      buffer = sentences[i];
    } else {
      buffer += " " + sentences[i];
    }
  }
  flush();
  return chunks;
}
