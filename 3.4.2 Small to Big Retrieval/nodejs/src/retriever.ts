/**
 * ============================================================================
 *  SMALL-TO-BIG (SENTENCE-WINDOW) RETRIEVAL  (Node.js / TypeScript)
 * ============================================================================
 *
 *  THE CORE PROBLEM
 *  ----------------
 *  The same chunking dilemma every RAG system faces:
 *
 *    Small chunks (~1 sentence):
 *      ✓ Tight, focused embeddings — great retrieval precision
 *      ✗ Too little surrounding context for the LLM to reason from
 *
 *    Large chunks (~a paragraph or more):
 *      ✓ Full context — the LLM can reason coherently
 *      ✗ Embedding is a blurry average of many ideas — retrieval suffers
 *
 *  SMALL-TO-BIG RESOLVES THIS — AND DIFFERS FROM PARENT-CHILD (3.4.1)
 *  -----------------------------------------------------------------
 *  Parent-child retrieval (3.4.1) splits each document into a FIXED set of
 *  parent sections at ingest time, sub-splits parents into children, and at
 *  query time maps every matched child back to its predefined parent.
 *
 *  Small-to-big retrieval takes a different, more granular tack:
 *
 *    • The atomic unit is a single SENTENCE (the "small" chunk).
 *    • Every sentence is embedded and indexed, tagged with its position
 *      (docId + ordinal) in the original document.
 *    • There are NO predefined parents. At query time, for each matched
 *      sentence we DYNAMICALLY expand a window of ±N neighbouring sentences
 *      around it — the "big" context — by walking the original sentence list.
 *    • Overlapping windows are MERGED so the LLM never sees a sentence twice.
 *
 *  In short:
 *    Parent-child  → static boundaries, map child → fixed parent.
 *    Small-to-big  → dynamic boundaries, expand sentence → sliding window,
 *                    then merge overlaps. The "big" chunk is centred exactly
 *                    on the hit, not snapped to a section boundary.
 *
 *  ARCHITECTURE
 *  ------------
 *
 *  INGESTION TIME:
 *    Document  (an ordered list of sentences)
 *       │
 *       ├── S0  ← embedded + indexed (pos 0)
 *       ├── S1  ← embedded + indexed (pos 1)
 *       ├── S2  ← embedded + indexed (pos 2)
 *       └── ...                       (the FULL ordered list is also kept,
 *                                      so we can expand windows at query time)
 *
 *  QUERY TIME (windowSize = 1):
 *    Query → embed → vector search (sentence index)
 *       → S5 scores 0.91
 *       → S6 scores 0.88
 *       → S12 scores 0.81
 *               │
 *               ▼
 *    Expand each hit to a window of ±1 sentence:
 *       → S5  → [S4, S5, S6]
 *       → S6  → [S5, S6, S7]
 *       → S12 → [S11, S12, S13]
 *               │
 *               ▼
 *    Merge overlapping/adjacent windows (S5 & S6 touch → [S4..S7]):
 *       → Window A: S4 S5 S6 S7   (from hits S5, S6)
 *       → Window B: S11 S12 S13   (from hit S12)
 *               │
 *               ▼
 *    Feed the merged windows to the LLM  ← contiguous context, no fragments,
 *                                           no repeated sentences.
 *
 *  DESIGN DECISIONS
 *  ----------------
 *  • The sentence index is the ONLY thing queried at search time.
 *  • Window expansion is bounded to the SAME document (no cross-doc bleed).
 *  • Merging is interval-union: any two windows that touch or overlap become
 *    one contiguous block, scored by the best hit inside it.
 *  • EmbedFn is an injectable callback — swap in OpenAI, Cohere, Voyage, or a
 *    mock without touching retrieval logic.
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// TYPES
// ---------------------------------------------------------------------------

/**
 * The atomic "small" unit — what actually gets embedded and searched.
 * It remembers where it lives so we can expand a window later.
 */
export interface Sentence {
  /** Unique identifier, e.g. "doc1-sent-7". */
  id: string;
  /** Which document this sentence belongs to. */
  docId: string;
  /** 0-based ordinal of this sentence within its document. */
  pos: number;
  /** The sentence text (the thing that gets embedded). */
  text: string;
}

/** A sentence paired with its cosine-similarity score from the vector search. */
export interface SentenceScore {
  sentence: Sentence;
  /** Cosine similarity score, range [0, 1] for normalised vectors. */
  score: number;
}

/**
 * The "big" context handed to the LLM: a contiguous block of sentences
 * expanded around one or more matched sentences, with overlapping windows
 * already merged.
 */
export interface WindowResult {
  docId: string;
  /** Position of the first sentence in the window. */
  startPos: number;
  /** Position of the last sentence in the window. */
  endPos: number;
  /** The joined sentence text — what the LLM reads. */
  text: string;
  /** Best hit score that contributed to this window. */
  bestScore: number;
  /** IDs of the sentence hits that fall inside this window. */
  matchedSentIds: string[];
}

/**
 * A function that converts text into a float vector (embedding).
 * Implement this with your real embedding provider.
 *
 * @example — with Voyage AI (recommended alongside Claude):
 *   import Anthropic from "@anthropic-ai/sdk";
 *   // Anthropic doesn't expose embeddings directly; use a compatible provider.
 *   // Voyage AI is the recommended partner:
 *   import VoyageAI from "voyageai";
 *   const client = new VoyageAI();
 *   const embedFn: EmbedFn = async (text) => {
 *     const res = await client.embed({ input: [text], model: "voyage-3" });
 *     return res.data[0].embedding;
 *   };
 */
export type EmbedFn = (text: string) => Promise<number[]>;

// ---------------------------------------------------------------------------
// IN-MEMORY STORE
// ---------------------------------------------------------------------------

/**
 * Holds the two data structures needed at query time:
 *   - sentences: the flat "vector index" of every sentence across all docs.
 *   - byDoc:     docId → the document's sentences in original order. This is
 *                what lets us expand a ±N window around any hit.
 *
 * In a production system:
 *   - sentences → pgvector table with an embedding column + HNSW index.
 *   - byDoc     → a plain Postgres table keyed by (doc_id, pos), no vectors.
 */
export interface SentenceStore {
  sentences: Sentence[];
  byDoc: Map<string, Sentence[]>;
}

/** Create a new empty SentenceStore. */
export function createStore(): SentenceStore {
  return { sentences: [], byDoc: new Map() };
}

// ---------------------------------------------------------------------------
// INGESTION
// ---------------------------------------------------------------------------

/**
 * Split `text` into trimmed, non-empty sentences.
 *
 * Splits on sentence-ending punctuation (. ! ?) followed by whitespace. This
 * is the minimal viable splitter; in production use a real sentence tokenizer
 * (e.g. an ICU segmenter) that handles abbreviations, decimals, and quotes.
 */
function splitIntoSentences(text: string): string[] {
  return text
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/**
 * Ingest a document into the store.
 *
 * The document is split into individual sentences. Each sentence is appended
 * to the flat search index AND recorded (in original order) under its docId so
 * windows can be expanded at query time.
 *
 * After calling this, embed every sentence and upsert those vectors into your
 * vector store before running any queries.
 *
 * @param store  The store to populate.
 * @param docId  A unique document identifier (namespaces all sentence IDs).
 * @param text   The raw document text.
 */
export function ingestDocument(
  store: SentenceStore,
  docId: string,
  text: string
): void {
  const sentTexts = splitIntoSentences(text);
  const docSents: Sentence[] = sentTexts.map((s, i) => ({
    id: `${docId}-sent-${i}`,
    docId,
    pos: i,
    text: s,
  }));
  store.sentences.push(...docSents);
  store.byDoc.set(docId, docSents);
}

// ---------------------------------------------------------------------------
// VECTOR SEARCH (sentence index)
// ---------------------------------------------------------------------------

/**
 * Compute cosine similarity between two equal-length vectors.
 * Returns a value in [−1, 1]; 1 = identical direction, 0 = orthogonal.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Search the sentence vector index and return the topK best matches,
 * sorted descending by cosine similarity score.
 *
 * In production this is a single SQL query against pgvector:
 *
 *   SELECT id, doc_id, pos, text,
 *          1 - (embedding <=> $queryVec) AS score
 *   FROM   sentences
 *   ORDER  BY embedding <=> $queryVec
 *   LIMIT  $topK;
 *
 * Here we embed all sentences in memory and compute scores manually.
 *
 * @param query  The raw user query string (gets embedded inside this function).
 * @param store  The store containing all ingested sentences.
 * @param embed  The embedding function to use.
 * @param topK   How many sentences to return.
 */
export async function searchSentences(
  query: string,
  store: SentenceStore,
  embed: EmbedFn,
  topK: number = 6
): Promise<SentenceScore[]> {
  const queryVec = await embed(query);

  const scored: SentenceScore[] = await Promise.all(
    store.sentences.map(async (sentence) => {
      const sentVec = await embed(sentence.text);
      return { sentence, score: cosineSimilarity(queryVec, sentVec) };
    })
  );

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// ---------------------------------------------------------------------------
// WINDOW EXPANSION + MERGE  (the "small → big" step)
// ---------------------------------------------------------------------------

/** Internal interval [start, end] of positions within a single document. */
interface Span {
  docId: string;
  start: number;
  end: number;
  bestScore: number;
  hitIds: string[];
}

/**
 * The heart of small-to-big retrieval.
 *
 *   1. For each sentence hit, expand to a window [pos-windowSize,
 *      pos+windowSize], clamped to the document's bounds.
 *   2. Group windows by document and sort by start position.
 *   3. Merge any windows that overlap OR are adjacent (touching) into a single
 *      contiguous span — an interval-union. The merged span's score is the
 *      best score among the hits it absorbed.
 *   4. Materialise each span into a WindowResult by joining the actual sentence
 *      text from store.byDoc.
 *
 * Results are returned sorted by bestScore descending, so the strongest match
 * leads the context fed to the LLM.
 *
 * @param store       The store to look up document sentences from.
 * @param hits        Top-k sentence hits from the vector search.
 * @param windowSize  Neighbouring sentences to include on EACH side of a hit.
 */
export function expandAndMerge(
  store: SentenceStore,
  hits: SentenceScore[],
  windowSize: number
): WindowResult[] {
  const w = windowSize < 0 ? 1 : windowSize;

  // Step 1: expand each hit into a clamped window span, grouped by document.
  const spansByDoc = new Map<string, Span[]>();
  for (const h of hits) {
    const docSents = store.byDoc.get(h.sentence.docId);
    if (!docSents) continue; // orphaned hit — skip gracefully

    const start = Math.max(0, h.sentence.pos - w);
    const end = Math.min(docSents.length - 1, h.sentence.pos + w);

    const list = spansByDoc.get(h.sentence.docId) ?? [];
    list.push({
      docId: h.sentence.docId,
      start,
      end,
      bestScore: h.score,
      hitIds: [h.sentence.id],
    });
    spansByDoc.set(h.sentence.docId, list);
  }

  // Steps 2 + 3: within each document, sort by start and merge overlaps.
  const merged: Span[] = [];
  for (const spans of spansByDoc.values()) {
    spans.sort((a, b) => a.start - b.start);
    let cur = spans[0];
    for (const s of spans.slice(1)) {
      // "+1" so windows that are merely adjacent (e.g. [4,6] and [7,9]) still
      // merge into one contiguous block.
      if (s.start <= cur.end + 1) {
        cur.end = Math.max(cur.end, s.end);
        cur.bestScore = Math.max(cur.bestScore, s.bestScore);
        cur.hitIds.push(...s.hitIds);
      } else {
        merged.push(cur);
        cur = s;
      }
    }
    merged.push(cur);
  }

  // Step 4: materialise spans into WindowResults with the joined text.
  const results: WindowResult[] = merged.map((s) => {
    const docSents = store.byDoc.get(s.docId)!;
    const texts: string[] = [];
    for (let p = s.start; p <= s.end; p++) texts.push(docSents[p].text);
    return {
      docId: s.docId,
      startPos: s.start,
      endPos: s.end,
      text: texts.join(" "),
      bestScore: s.bestScore,
      matchedSentIds: s.hitIds,
    };
  });

  // Order by best score descending — strongest context first.
  return results.sort((a, b) => b.bestScore - a.bestScore);
}

// ---------------------------------------------------------------------------
// PUBLIC RETRIEVAL PIPELINE
// ---------------------------------------------------------------------------

/**
 * Complete small-to-big retrieval pipeline.
 *
 * 1. Embed the query using `embed`.
 * 2. Search the sentence vector index for the `topKSentences` most similar.
 * 3. Expand each matched sentence to a ±windowSize window of neighbours.
 * 4. Merge overlapping/adjacent windows into contiguous blocks.
 * 5. Return the merged windows ordered by best sentence similarity score.
 *
 * The caller passes each window's `text` directly to the LLM as context.
 *
 * @param query          Raw user question.
 * @param store          Populated SentenceStore.
 * @param embed          Embedding function (real or mock).
 * @param topKSentences  How many sentences to retrieve before expansion.
 * @param windowSize     Neighbouring sentences on EACH side of a hit
 *                       (1 → 3-sentence windows, 2 → 5-sentence, …).
 * @param maxWindows     Upper bound on returned windows (0 = no limit).
 */
export async function retrieve(
  query: string,
  store: SentenceStore,
  embed: EmbedFn,
  topKSentences: number = 6,
  windowSize: number = 1,
  maxWindows: number = 0
): Promise<WindowResult[]> {
  // Step 1 + 2: embed query → search sentence index.
  const hits = await searchSentences(query, store, embed, topKSentences);

  // Step 3 + 4: expand each hit to a window, then merge overlaps.
  let results = expandAndMerge(store, hits, windowSize);

  // Step 5: apply maxWindows cap.
  if (maxWindows > 0) {
    results = results.slice(0, maxWindows);
  }

  return results;
}
