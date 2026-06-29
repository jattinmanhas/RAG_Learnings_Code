/**
 * ============================================================================
 *  SMALL-TO-BIG RETRIEVAL DEMO  —  run with:  npm start
 * ============================================================================
 *
 *  Demonstrates the complete small-to-big (sentence-window) retrieval pipeline
 *  on a small in-memory sentence store. No API key or network required — the
 *  mock embedding function produces deterministic vectors based on keyword
 *  counts, which is enough to show the retrieval logic working correctly.
 *
 *  WHAT YOU WILL SEE
 *  -----------------
 *    Ingestion phase:   documents → sentences (the "small" chunks)
 *    Query 1:           HNSW vs IVFFlat question, windowSize = 1
 *                       → adjacent sentence hits merge into one window
 *    Query 2:           same query, windowSize = 2 → wider merged block
 *    Query 3:           chunking-overlap question on the second document
 *    Merge demo:        side-by-side count of raw hits vs merged windows
 *
 *  REPLACING THE MOCK
 *  ------------------
 *  Replace `mockEmbed` with a real embedding call, e.g. with Voyage AI
 *  (the recommended embedding partner for Claude-based RAG pipelines):
 *
 *    import VoyageAI from "voyageai";
 *    const voyage = new VoyageAI({ apiKey: process.env.VOYAGE_API_KEY });
 *    const embedFn: EmbedFn = async (text) => {
 *      const res = await voyage.embed({ input: [text], model: "voyage-3" });
 *      return res.data[0].embedding;
 *    };
 *
 *  And replace the in-memory store with pgvector (a sentences table with an
 *  embedding column + HNSW index, plus a plain (doc_id, pos) table for
 *  window expansion).
 * ============================================================================
 */

import {
  createStore,
  ingestDocument,
  searchSentences,
  retrieve,
  EmbedFn,
  SentenceScore,
  WindowResult,
} from "./retriever";

// ---------------------------------------------------------------------------
// SAMPLE DOCUMENTS
// ---------------------------------------------------------------------------
// Unlike parent-child (3.4.1), small-to-big does NOT pre-split documents into
// sections. Each document is continuous prose; ingestDocument breaks it into
// individual SENTENCES, and window expansion at query time decides how much
// surrounding context to pull back.
//
// Two documents are used so you can see window expansion is bounded to the
// same document (no cross-document bleed).

const DOCUMENTS: Record<string, string> = {
  indexing: `HNSW is the most widely deployed approximate nearest-neighbour index for dense vector search. It builds a multi-layer proximity graph where each node is a vector. Queries traverse the graph from a random entry point at the top layer and greedily descend to denser layers. HNSW achieves sub-linear query time and can be updated incrementally without a rebuild. IVFFlat instead partitions the vector space into Voronoi cells using k-means clustering. At query time only the nprobe closest centroids are searched, reducing the comparison set dramatically. Unlike HNSW, IVFFlat requires a full rebuild when nlist changes and cannot be updated incrementally. HNSW consumes more memory but supports real-time inserts and higher recall. IVFFlat uses less memory and is faster to build on a large static corpus. For most production RAG workloads with ongoing ingestion, HNSW is the safer default.`,

  chunking: `Fixed-size chunking divides a document into non-overlapping windows of N tokens. It is the simplest strategy to implement and produces uniform index sizes. The main risk is boundary cuts that split a sentence mid-clause and lose meaning. Semantic chunking instead groups sentences by measuring the embedding distance between consecutive sentences. When the distance spikes above a threshold, a new chunk begins. This keeps topically coherent passages together at the cost of variable chunk sizes. Overlap is a complementary technique where a sliding window shares tokens between adjacent chunks. Overlap ensures information near a boundary appears in at least two chunks. A twenty percent overlap is a common starting point but increases index size proportionally.`,
};

// Stable iteration order for printing.
const DOC_ORDER = ["indexing", "chunking"];

// ---------------------------------------------------------------------------
// MOCK EMBEDDING FUNCTION
// ---------------------------------------------------------------------------
// Maps any text to a 6-dimensional "concept vector" by counting keyword
// matches across six semantic dimensions. L2-normalised so cosine similarity
// equals the dot product.
//
// Dimension legend:
//   [0] HNSW / graph-index concepts
//   [1] IVFFlat / centroid / Voronoi concepts
//   [2] memory / recall / build trade-off concepts
//   [3] fixed-size / boundary chunking concepts
//   [4] semantic chunking / distance-threshold concepts
//   [5] overlap / sliding window concepts

const CONCEPT_KEYWORDS: string[][] = [
  ["hnsw", "graph", "navigable", "layer", "node", "greedy", "descend", "entry", "proximity"],
  ["ivfflat", "ivf", "voronoi", "centroid", "cluster", "nprobe", "nlist", "kmeans", "k-means"],
  ["memory", "recall", "rebuild", "incremental", "insert", "build", "corpus", "ram"],
  ["fixed", "boundary", "uniform", "cut", "clause", "non-overlapping", "token"],
  ["semantic", "distance", "threshold", "coherent", "consecutive", "spike", "topically"],
  ["overlap", "stride", "sliding", "adjacent", "shared", "twenty", "percent", "proportional"],
];

function l2Normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  if (norm === 0) return v;
  return v.map((x) => x / norm);
}

const mockEmbed: EmbedFn = async (text: string): Promise<number[]> => {
  const lower = text.toLowerCase();
  const vec = CONCEPT_KEYWORDS.map((keywords) =>
    keywords.reduce((count, kw) => {
      let n = 0, pos = 0;
      while ((pos = lower.indexOf(kw, pos)) !== -1) { n++; pos += kw.length; }
      return count + n;
    }, 0)
  );
  return l2Normalize(vec);
};

// ---------------------------------------------------------------------------
// DISPLAY HELPERS
// ---------------------------------------------------------------------------

const HR = "=".repeat(76);

function header(title: string): void {
  console.log(`\n${HR}\n${title}\n${HR}`);
}

function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen) + "…";
}

function printStore(store: ReturnType<typeof createStore>): void {
  console.log(
    `\n  Ingested ${store.byDoc.size} documents, ${store.sentences.length} sentences total.\n`
  );
  for (const docId of DOC_ORDER) {
    const sents = store.byDoc.get(docId)!;
    console.log(`  doc "${docId}" — ${sents.length} sentences:`);
    sents.forEach((s) => {
      console.log(`    [${s.pos}] ${s.id.padEnd(18)} ${JSON.stringify(truncate(s.text, 62))}`);
    });
    console.log();
  }
}

function printHits(hits: SentenceScore[]): void {
  hits.forEach((h, i) => {
    console.log(
      `    [${i}] score=${h.score.toFixed(4)}  ${h.sentence.id} ` +
      `(doc=${h.sentence.docId} pos=${h.sentence.pos})`
    );
    console.log(`         ${JSON.stringify(truncate(h.sentence.text, 68))}`);
  });
}

function printWindows(results: WindowResult[]): void {
  if (results.length === 0) {
    console.log("  (no results)");
    return;
  }
  results.forEach((r, i) => {
    console.log(
      `  [${i}] doc=${r.docId}  positions ${r.startPos}..${r.endPos} ` +
      `(${r.endPos - r.startPos + 1} sentences)`
    );
    console.log(`       bestScore       = ${r.bestScore.toFixed(4)}`);
    console.log(`       matchedSentIds  = ${JSON.stringify(r.matchedSentIds)}`);
    console.log(`       context         → ${JSON.stringify(truncate(r.text, 160))}\n`);
  });
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  // -----------------------------------------------------------------------
  // INGESTION PHASE
  // -----------------------------------------------------------------------
  header("INGESTION PHASE — split documents into sentences");

  const store = createStore();
  for (const docId of DOC_ORDER) {
    ingestDocument(store, docId, DOCUMENTS[docId]);
  }
  printStore(store);

  // -----------------------------------------------------------------------
  // QUERY 1 — targets the "indexing" document (windowSize = 1)
  // -----------------------------------------------------------------------
  header("QUERY 1 — HNSW vs IVFFlat trade-offs (windowSize = 1)");

  const q1 = "How does HNSW differ from IVFFlat and which uses more memory?";
  console.log(`  Query: "${q1}"\n`);

  const hits1 = await searchSentences(q1, store, mockEmbed, 4);
  console.log("  Step 1 — Top-4 sentence hits (the 'small' chunks):");
  printHits(hits1);

  console.log("\n  Step 2 — Expand ±1 sentence and merge overlaps (the 'big' chunks):");
  const results1 = await retrieve(q1, store, mockEmbed, 4, 1);
  printWindows(results1);

  // -----------------------------------------------------------------------
  // QUERY 2 — same query, wider window
  // -----------------------------------------------------------------------
  header("QUERY 2 — Same query, windowSize = 2 (wider context)");

  console.log(`  Query: "${q1}"\n`);
  const results2 = await retrieve(q1, store, mockEmbed, 4, 2);
  console.log("  Wider windows → fewer, larger merged blocks:");
  printWindows(results2);

  // -----------------------------------------------------------------------
  // QUERY 3 — targets the "chunking" document
  // -----------------------------------------------------------------------
  header("QUERY 3 — Chunking overlap (windowSize = 1)");

  const q3 = "What is overlap in chunking and how does it affect index size?";
  console.log(`  Query: "${q3}"\n`);

  const hits3 = await searchSentences(q3, store, mockEmbed, 3);
  console.log("  Step 1 — Top-3 sentence hits:");
  printHits(hits3);

  console.log("\n  Step 2 — Expand ±1 sentence and merge overlaps:");
  const results3 = await retrieve(q3, store, mockEmbed, 3, 1);
  printWindows(results3);

  // -----------------------------------------------------------------------
  // MERGE DEMO — why interval-union matters
  // -----------------------------------------------------------------------
  header("WINDOW MERGE — why it matters");

  console.log("  Query 1 produced these raw sentence hits:");
  hits1.forEach((h) => {
    console.log(`    ${h.sentence.id} (doc=${h.sentence.docId} pos=${h.sentence.pos})`);
  });
  console.log(`\n  Without merge: ${hits1.length} separate ±1 windows (overlapping, repeating sentences)`);
  console.log(`  With    merge: ${results1.length} contiguous block(s) — no sentence repeated`);
  console.log("\n  → Adjacent hits collapse into one clean window. The LLM reads a");
  console.log("    continuous passage centred on the match, not snapped to a fixed");
  console.log("    section boundary (that's the difference from parent-child, 3.4.1).");

  console.log(`\n${HR}`);
  console.log("Done. Replace mockEmbed with a real provider (Voyage AI, Cohere, OpenAI, …)");
  console.log("and replace the in-memory store with pgvector (sentences table with an");
  console.log("embedding column + a plain (doc_id, pos) table for window expansion).");
  console.log(HR);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
