/**
 * ============================================================================
 *  PARENT-CHILD RETRIEVAL  (Node.js / TypeScript)
 * ============================================================================
 *
 *  THE CORE PROBLEM
 *  ----------------
 *  Every RAG system faces a chunking dilemma:
 *
 *    Small chunks (~150 tokens):
 *      ✓ Tight, focused embeddings — great retrieval precision
 *      ✗ Too little surrounding context for the LLM to reason from
 *
 *    Large chunks (~800 tokens):
 *      ✓ Full section context — the LLM can reason coherently
 *      ✗ Embedding is a blurry average of many sub-topics — retrieval suffers
 *
 *  PARENT-CHILD RETRIEVAL RESOLVES THIS BY SEPARATING THE CONCERN:
 *    SEARCH  with child embeddings  → precise vector match
 *    FEED    parent chunks to LLM  → full reasoning context
 *
 *  ARCHITECTURE
 *  ------------
 *
 *  INGESTION TIME:
 *    Document
 *       │
 *       ├── Parent 1 (full section, ~800 tokens)  ← stored in a map, NOT embedded
 *       │       ├── Child 1.1 (~150 tokens)  ← embedded + indexed in vector store
 *       │       ├── Child 1.2 (~150 tokens)  ← embedded + indexed in vector store
 *       │       └── Child 1.3 (~150 tokens)  ← embedded + indexed in vector store
 *       │
 *       ├── Parent 2 (full section, ~700 tokens)  ← stored, NOT embedded
 *       │       ├── Child 2.1 (~150 tokens)  ← embedded + indexed
 *       │       ├── Child 2.2 (~150 tokens)  ← embedded + indexed
 *       │       └── Child 2.3 (~150 tokens)  ← embedded + indexed
 *       └── ...
 *
 *  QUERY TIME:
 *    Query → embed → vector search (child index only)
 *       → Child 2.1 scores 0.91
 *       → Child 2.2 scores 0.87
 *       → Child 1.3 scores 0.82
 *               │
 *               ▼
 *    Fetch parent of each match
 *       → Parent 2  (from Child 2.1 and 2.2)
 *       → Parent 1  (from Child 1.3)
 *               │
 *               ▼
 *    Deduplicate by parentId  ← key step: 2.1 and 2.2 both point to Parent 2,
 *                                          returned only ONCE
 *               │
 *               ▼
 *    Feed Parent 1 + Parent 2 to LLM  ← full context, no fragments
 *
 *  DESIGN DECISIONS
 *  ----------------
 *  • The child index is the ONLY thing queried at search time.
 *    Parents are never put in the vector store.
 *  • Parent deduplication is stable: if N children from the same parent all
 *    match, the parent appears once, ordered by the best child score.
 *  • EmbedFn is an injectable callback — swap in OpenAI, Cohere, or a mock
 *    without touching retrieval logic.
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// TYPES
// ---------------------------------------------------------------------------

/**
 * A large section of a document — what the LLM ultimately reads as context.
 * Stored in a key-value map; NEVER embedded or added to the vector index.
 */
export interface ParentChunk {
  /** Unique identifier, e.g. "doc1-parent-0". */
  id: string;
  /** Section heading, if extractable from the text. */
  title: string;
  /** Full section text (~700-900 tokens in production). */
  text: string;
}

/**
 * A short passage derived from a parent chunk.
 * ONLY children are embedded and indexed in the vector store.
 * The parentId field is the foreign key that links back to the parent.
 */
export interface ChildChunk {
  /** Unique identifier, e.g. "doc1-parent-0-child-1". */
  id: string;
  /** Foreign key — which parent this child was derived from. */
  parentId: string;
  /** The short passage that gets embedded (~100-200 tokens in production). */
  text: string;
}

/** A child chunk paired with its similarity score from the vector search. */
export interface ChildScore {
  child: ChildChunk;
  /** Cosine similarity score, range [0, 1] for normalised vectors. */
  score: number;
}

/**
 * What the retrieval pipeline hands to the LLM.
 * Contains the full parent text plus diagnostic info about which children
 * triggered the match and how confident those matches were.
 */
export interface ParentResult {
  parent: ParentChunk;
  /** Highest similarity score among all children that matched this parent. */
  bestChildScore: number;
  /** IDs of the child chunks that pointed to this parent. */
  matchedChildren: string[];
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
 *   - parents:  Map<parentId, ParentChunk>  (key-value lookup — no vectors)
 *   - children: ChildChunk[]                (the "vector index")
 *
 * In a production system:
 *   - parents  → Redis, DynamoDB, or a plain Postgres table (no vector column)
 *   - children → pgvector table with an HNSW index on the embedding column
 */
export interface ParentChildStore {
  parents: Map<string, ParentChunk>;
  children: ChildChunk[];
}

/** Create a new empty ParentChildStore. */
export function createStore(): ParentChildStore {
  return { parents: new Map(), children: [] };
}

// ---------------------------------------------------------------------------
// INGESTION
// ---------------------------------------------------------------------------

/**
 * Split `text` into passages of at most `maxWords` words each.
 *
 * This is a word-boundary splitter — the minimal viable implementation.
 * In production use the sentence-aware or semantic splitter from chapter 3.2
 * so that child boundaries respect sentence structure.
 */
function splitIntoChildren(text: string, maxWords: number): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += maxWords) {
    chunks.push(words.slice(i, i + maxWords).join(" "));
  }
  return chunks;
}

/**
 * Attempt to extract a section title from the first line of `text`.
 * A line is treated as a title if it is short (<80 chars) and doesn't end
 * with a period (i.e. it looks like a heading, not a sentence).
 */
function extractTitle(text: string): string {
  const firstLine = text.trimStart().split("\n")[0].trim();
  if (firstLine.length < 80 && !firstLine.endsWith(".")) {
    return firstLine;
  }
  return "";
}

/**
 * Ingest a document into the store.
 *
 * Each element in `parentTexts` becomes one parent chunk.
 * Each parent is then sub-split into child chunks of ~`childSizeWords` words.
 *
 * After calling this, embed every child and upsert those vectors into your
 * vector store before running any queries.
 *
 * @param store         The store to populate.
 * @param docId         A unique document identifier (namespaces all chunk IDs).
 * @param parentTexts   Pre-split section texts that become parent chunks.
 *                      In production, these come from a semantic section splitter.
 * @param childSizeWords  Approximate word count per child chunk.
 */
export function ingestDocument(
  store: ParentChildStore,
  docId: string,
  parentTexts: string[],
  childSizeWords: number = 100
): void {
  parentTexts.forEach((pText, pIdx) => {
    const parentId = `${docId}-parent-${pIdx}`;

    // Store the full parent — this is what the LLM will read.
    store.parents.set(parentId, {
      id: parentId,
      title: extractTitle(pText),
      text: pText,
    });

    // Sub-split into children for embedding.
    const childTexts = splitIntoChildren(pText, childSizeWords);
    childTexts.forEach((cText, cIdx) => {
      store.children.push({
        id: `${parentId}-child-${cIdx}`,
        parentId,
        text: cText,
      });
    });
  });
}

// ---------------------------------------------------------------------------
// VECTOR SEARCH (child index)
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
 * Search the child vector index and return the topK best matches,
 * sorted descending by cosine similarity score.
 *
 * In production this is a single SQL query against pgvector:
 *
 *   SELECT id, parent_id, text,
 *          1 - (embedding <=> $queryVec) AS score
 *   FROM   child_chunks
 *   ORDER  BY embedding <=> $queryVec
 *   LIMIT  $topK;
 *
 * Here we embed all children in memory and compute scores manually.
 *
 * @param query  The raw user query string (gets embedded inside this function).
 * @param store  The store containing all ingested child chunks.
 * @param embed  The embedding function to use.
 * @param topK   How many children to return.
 */
export async function searchChildren(
  query: string,
  store: ParentChildStore,
  embed: EmbedFn,
  topK: number = 6
): Promise<ChildScore[]> {
  // Embed the query.
  const queryVec = await embed(query);

  // Score every child against the query embedding.
  const scored: ChildScore[] = await Promise.all(
    store.children.map(async (child) => {
      const childVec = await embed(child.text);
      return { child, score: cosineSimilarity(queryVec, childVec) };
    })
  );

  // Sort descending by score and return the top K.
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ---------------------------------------------------------------------------
// PARENT FETCH + DEDUPLICATION
// ---------------------------------------------------------------------------

/**
 * Map child scores → their parent chunks, deduplicating by parentId.
 *
 * This is THE key step of parent-child retrieval:
 *   - Multiple children may point to the same parent.
 *   - We return that parent exactly once.
 *   - We record ALL matching child IDs and the BEST (highest) child score.
 *   - Output is ordered by first-appearance, which is equivalent to
 *     best-score-first because `childScores` is already sorted descending.
 *
 * @param store        The store to look up parents from.
 * @param childScores  Top-k children from the vector search, sorted desc.
 */
export function fetchAndDedup(
  store: ParentChildStore,
  childScores: ChildScore[]
): ParentResult[] {
  const seen = new Map<string, ParentResult>();
  const order: string[] = []; // preserves insertion order (= best-score-first)

  for (const cs of childScores) {
    const { parentId } = cs.child;
    const parent = store.parents.get(parentId);
    if (!parent) continue; // orphaned child — skip gracefully

    const existing = seen.get(parentId);
    if (existing) {
      // Parent already in the result set — just add the new child ID.
      // BestChildScore stays as-is: first encounter = best score (desc sorted).
      existing.matchedChildren.push(cs.child.id);
    } else {
      const result: ParentResult = {
        parent,
        bestChildScore: cs.score,
        matchedChildren: [cs.child.id],
      };
      seen.set(parentId, result);
      order.push(parentId);
    }
  }

  return order.map((pid) => seen.get(pid)!);
}

// ---------------------------------------------------------------------------
// PUBLIC RETRIEVAL PIPELINE
// ---------------------------------------------------------------------------

/**
 * Complete parent-child retrieval pipeline.
 *
 * 1. Embed the query using `embed`.
 * 2. Search the child vector index for the `topKChildren` most similar children.
 * 3. Look up the parent of every matched child.
 * 4. Deduplicate parents (multiple children → same parent → one entry).
 * 5. Return parent chunks ordered by best child similarity score.
 *
 * The caller passes the returned parent texts directly to the LLM as context.
 *
 * @param query         Raw user question.
 * @param store         Populated ParentChildStore.
 * @param embed         Embedding function (real or mock).
 * @param topKChildren  How many children to retrieve before parent-mapping.
 *                      More children = higher recall, more parents returned.
 * @param maxParents    Upper bound on returned parents (0 = no limit).
 */
export async function retrieve(
  query: string,
  store: ParentChildStore,
  embed: EmbedFn,
  topKChildren: number = 6,
  maxParents: number = 0
): Promise<ParentResult[]> {
  // Step 1 + 2: embed query → search child index.
  const childScores = await searchChildren(query, store, embed, topKChildren);

  // Step 3 + 4: map children → parents, deduplicate.
  let results = fetchAndDedup(store, childScores);

  // Step 5: apply maxParents cap.
  if (maxParents > 0) {
    results = results.slice(0, maxParents);
  }

  return results;
}
