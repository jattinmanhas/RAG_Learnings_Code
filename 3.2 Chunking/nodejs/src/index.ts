/**
 * ============================================================================
 *  CHUNKING DEMO  —  run with:  npm start
 * ============================================================================
 *
 *  This runs every strategy from ./chunkers.ts on the same sample text so you
 *  can SEE how they differ. Read the output side-by-side with the comments in
 *  chunkers.ts.
 *
 *  CHEAT SHEET — which strategy should I reach for?
 *  ------------------------------------------------
 *    Fixed-size .............. quick baseline; unstructured text (logs, OCR).
 *    Fixed + overlap ......... default for fixed approaches; stops boundary loss.
 *    Sentence ................ well-punctuated prose; want whole thoughts.
 *    Paragraph ............... blank-line-structured docs; respect author intent.
 *    Recursive ⭐ ............ BEST general default for mixed/unknown content.
 *    Token ................... must respect a hard token limit / control cost.
 *    Markdown-header ......... docs/wikis with headings; want section metadata.
 *    Semantic ................ high-value corpora; topic-drifting text; costlier.
 * ============================================================================
 */

import {
  Chunk,
  EmbedFn,
  fixedSizeChunk,
  fixedSizeChunkWithOverlap,
  sentenceChunk,
  paragraphChunk,
  recursiveChunk,
  tokenChunk,
  markdownHeaderChunk,
  semanticChunk,
} from "./chunkers";

const SAMPLE = `# Retrieval-Augmented Generation

RAG combines a retriever with a generator. The retriever finds relevant text. The generator writes an answer grounded in that text.

## Why Chunking Matters

Embedding models have token limits. You cannot embed a whole book at once. Smaller chunks also make retrieval sharper, because each vector represents one focused idea instead of a blurry average.

## Trade-offs

Chunks that are too small lose context. Chunks that are too large waste the context window and dilute relevance. Overlap helps facts survive across boundaries.`;

function preview(text: string, n = 60): string {
  const flat = text.replace(/\s+/g, " ");
  return flat.length > n ? flat.slice(0, n) + "…" : flat;
}

function report(title: string, chunks: Chunk[]): void {
  console.log(`\n${"=".repeat(76)}\n${title}\n${"=".repeat(76)}`);
  console.log(`  ${chunks.length} chunk(s):`);
  for (const c of chunks) {
    const meta = c.metadata ? `  ${JSON.stringify(c.metadata)}` : "";
    console.log(`  [${c.index}] (${c.text.length} chars) "${preview(c.text)}"${meta}`);
  }
}

/**
 * A FAKE embedding function so the semantic demo runs with no API key / deps.
 * It maps text to a tiny vector based on keyword presence, which is enough to
 * make the topic-boundary logic visible. Replace with a real embedder in prod.
 */
const fakeEmbed: EmbedFn = async (texts) =>
  texts.map((t) => {
    const lower = t.toLowerCase();
    const topics = ["retriev", "embed", "chunk", "context", "overlap", "answer"];
    return topics.map((kw) => (lower.includes(kw) ? 1 : 0));
  });

async function main() {
  report("1. FIXED-SIZE (size=200)", fixedSizeChunk(SAMPLE, 200));
  report("2. FIXED-SIZE + OVERLAP (size=200, overlap=40)", fixedSizeChunkWithOverlap(SAMPLE, 200, 40));
  report("3. SENTENCE (maxChars=200)", sentenceChunk(SAMPLE, 200));
  report("4. PARAGRAPH (maxChars=300)", paragraphChunk(SAMPLE, 300));
  report("5. RECURSIVE ⭐ (maxChars=200, overlap=30)", recursiveChunk(SAMPLE, 200, 30));
  report("6. TOKEN (maxTokens=60, overlap=10)", tokenChunk(SAMPLE, 60, 10));
  report("7. MARKDOWN-HEADER (maxChars=400)", markdownHeaderChunk(SAMPLE, 400));
  report("8. SEMANTIC (threshold=0.75)", await semanticChunk(SAMPLE, fakeEmbed, { similarityThreshold: 0.75, maxChars: 400 }));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
