/**
 * ============================================================================
 *  PARENT-CHILD RETRIEVAL DEMO  —  run with:  npm start
 * ============================================================================
 *
 *  Demonstrates the complete parent-child retrieval pipeline on a small
 *  in-memory document store.  No API key or network required — the mock
 *  embedding function produces deterministic vectors based on keyword counts,
 *  which is enough to show the retrieval logic working correctly.
 *
 *  WHAT YOU WILL SEE
 *  -----------------
 *    Ingestion phase:   document → parent chunks → child chunks (list)
 *    Query 1:           HNSW vs IVFFlat question → children from Parent 0
 *                       → deduplication collapses 4 children to 1 parent
 *    Query 2:           Embedding dimensionality question → children from Parent 1
 *    Query 3:           Cross-cutting question → children from Parents 0, 1, 2
 *                       → three distinct parents returned
 *    Dedup demo:        side-by-side count of "without dedup" vs "with dedup"
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
 *  And replace the in-memory store with pgvector (parents in a plain table,
 *  children in a table with an embedding column and HNSW index).
 * ============================================================================
 */

import {
  createStore,
  ingestDocument,
  searchChildren,
  fetchAndDedup,
  retrieve,
  EmbedFn,
  ChildScore,
  ParentResult,
} from "./retriever";

// ---------------------------------------------------------------------------
// SAMPLE DOCUMENT
// ---------------------------------------------------------------------------
// A three-section technical document about RAG internals.
// Each section → one parent chunk.
// Each paragraph within a section → one child chunk.
//
// Sizes here are ~150-200 words/parent (≈200-270 tokens) for demo compactness.
// In production: parents ~800 tokens, children ~150 tokens.

const PARENT_TEXTS = [
  // Section 0 — Vector Indexing Strategies
  `Vector Indexing Strategies

HNSW (Hierarchical Navigable Small World) is the most widely deployed approximate nearest-neighbour index for dense vector search. It builds a multi-layer proximity graph where each node is a vector. Queries traverse the graph from a random entry point at the top (sparse) layer and greedily descend to denser layers until the nearest neighbours are identified. HNSW achieves sub-linear query time — typically O(log n) — and can be updated incrementally without a rebuild.

IVFFlat (Inverted File with Flat quantization) partitions the vector space into Voronoi cells using k-means clustering. At index time each vector is assigned to its nearest centroid. At query time only the nprobe closest centroids are searched, reducing the comparison set from n to roughly n/nlist vectors. Unlike HNSW, IVFFlat requires a full rebuild when nlist changes and cannot be updated incrementally.

Choosing between HNSW and IVFFlat depends on your write pattern and memory budget. HNSW consumes more RAM (storing the graph edges) but supports real-time inserts and delivers higher recall at the same latency budget. IVFFlat uses less memory and is faster to build on a large static corpus, but recall drops sharply if nprobe is set too low. For most production RAG workloads with ongoing ingestion, HNSW is the safer default.`,

  // Section 1 — Embedding Model Selection
  `Embedding Model Selection

Dense embedding models map text to a fixed-length float vector in a high-dimensional space (typically 768 to 3072 dimensions). Models such as text-embedding-3-large (OpenAI), embed-english-v3.0 (Cohere), and voyage-3 (Voyage AI) are trained on large corpora with contrastive objectives — they pull semantically similar passages close together and push dissimilar ones apart. The choice of model affects both retrieval quality and operational cost: larger models have higher dimensionality and slower throughput.

Sparse representations like BM25 or SPLADE score term frequency rather than semantic proximity. They excel at keyword-critical queries (product codes, proper nouns, rare terms) where dense models hallucinate similarity. Hybrid retrieval — running a dense retriever and a sparse retriever in parallel, then fusing their ranked lists with Reciprocal Rank Fusion — captures the strengths of both and consistently outperforms either alone on heterogeneous corpora.

Embedding dimensionality and normalization matter more than most practitioners expect. Truncating OpenAI's text-embedding-3-large from 3072 to 1536 dimensions loses less than 2 % NDCG@10 on MTEB while halving storage and index build time. Always L2-normalise embeddings before indexing when using cosine similarity — without normalisation, dot-product search and cosine search give different rankings, and most vector stores assume normalised inputs.`,

  // Section 2 — Chunking Strategies
  `Chunking Strategies for RAG

Fixed-size chunking divides a document into non-overlapping windows of N tokens, ignoring sentence or paragraph boundaries. It is the simplest strategy to implement and produces uniform index sizes. The main risk is boundary cuts: a sentence split mid-clause loses meaning in both resulting chunks, degrading embedding quality. Fixed-size chunking is an acceptable baseline when documents are already structured (tables, code blocks, enumerated lists).

Semantic chunking groups sentences into chunks by measuring the embedding distance between consecutive sentences. When the distance spikes above a threshold, a new chunk begins. This keeps topically coherent passages together and avoids mid-topic splits, at the cost of variable chunk sizes. Variable sizes complicate batching for embedding APIs that charge per token, but the retrieval quality gain usually justifies it.

Overlap is a complementary technique: a sliding window moves forward by a stride smaller than the window size, so adjacent chunks share N tokens of context. Overlap ensures that information near a boundary appears in at least two chunks, reducing the chance of a critical sentence being isolated. A 20 % overlap (e.g. 20 shared tokens in a 100-token window) is a common starting point. Overlap increases index size proportionally to the overlap fraction, so it must be balanced against storage cost.`,
];

// ---------------------------------------------------------------------------
// MOCK EMBEDDING FUNCTION
// ---------------------------------------------------------------------------
// Maps any text string to a 6-dimensional "concept vector" by counting keyword
// matches across six semantic dimensions.  L2-normalised so cosine similarity
// equals the dot product.
//
// Dimension legend:
//   [0] HNSW / graph-index concepts
//   [1] IVFFlat / centroid / Voronoi concepts
//   [2] embedding model / dimensionality concepts
//   [3] sparse / BM25 / hybrid retrieval concepts
//   [4] chunking / splitting / boundary concepts
//   [5] overlap / sliding window concepts

const CONCEPT_KEYWORDS: string[][] = [
  ["hnsw", "graph", "hierarchical", "navigable", "layer", "node", "greedy", "descend", "entry"],
  ["ivfflat", "ivf", "voronoi", "centroid", "cluster", "nprobe", "nlist", "kmeans", "k-means"],
  ["embedding", "model", "dimension", "vector", "dense", "normalise", "normalize", "l2", "truncat"],
  ["sparse", "bm25", "splade", "hybrid", "keyword", "term", "frequency", "fusion", "reciprocal"],
  ["chunk", "chunking", "split", "fixed", "semantic", "boundary", "sentence", "window", "token"],
  ["overlap", "stride", "sliding", "adjacent", "shared", "20%", "20 %", "proportional"],
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
      // Count all non-overlapping occurrences of kw in lower.
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

function printStore(parents: Map<string, unknown>, children: { id: string; parentId: string; text: string }[]): void {
  console.log(`\n  Ingested ${parents.size} parent chunks, ${children.length} child chunks total.\n`);
  children.forEach((c, i) => {
    console.log(`  child[${i}]  id=${c.id.padEnd(34)}  parent=${c.parentId}`);
    console.log(`            text=${JSON.stringify(truncate(c.text, 80))}\n`);
  });
}

function printChildScores(scores: ChildScore[]): void {
  scores.forEach((cs, i) => {
    console.log(
      `    [${i}] score=${cs.score.toFixed(4)}  childId=${cs.child.id.padEnd(34)}` +
      `  text=${JSON.stringify(truncate(cs.child.text, 55))}`
    );
  });
}

function printResults(results: ParentResult[]): void {
  if (results.length === 0) {
    console.log("  (no results)");
    return;
  }
  results.forEach((r, i) => {
    console.log(`  [${i}] parentId      = ${r.parent.id}`);
    console.log(`       title         = ${JSON.stringify(r.parent.title)}`);
    console.log(`       bestScore     = ${r.bestChildScore.toFixed(4)}`);
    console.log(`       matchedChildren = ${JSON.stringify(r.matchedChildren)}`);
    console.log(`       context       → ${JSON.stringify(truncate(r.parent.text, 120))}\n`);
  });
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  // -----------------------------------------------------------------------
  // INGESTION PHASE
  // -----------------------------------------------------------------------
  header("INGESTION PHASE — split document into parents and children");

  const store = createStore();
  // childSizeWords = 50 words/child (≈65 tokens) — compact for demo.
  // In production: parentSize ~800 tokens, childSizeWords ~115 words (~150 tokens).
  ingestDocument(store, "rag-guide", PARENT_TEXTS, 50);
  printStore(store.parents, store.children);

  // -----------------------------------------------------------------------
  // QUERY 1 — targets Parent 0 (Vector Indexing)
  // -----------------------------------------------------------------------
  header("QUERY 1 — Vector index trade-offs");

  const q1 = "What are the differences between HNSW and IVFFlat for approximate nearest-neighbour search?";
  console.log(`  Query: "${q1}"\n`);

  // Show which children are retrieved BEFORE parent mapping.
  const childHits1 = await searchChildren(q1, store, mockEmbed, 4);
  console.log("  Step 1 — Top-4 children from vector search:");
  printChildScores(childHits1);

  // Full pipeline — map children → parents → deduplicate.
  const results1 = fetchAndDedup(store, childHits1);
  console.log("\n  Step 2 — Fetch parents, deduplicate:");
  printResults(results1);

  // -----------------------------------------------------------------------
  // QUERY 2 — targets Parent 1 (Embedding Models)
  // -----------------------------------------------------------------------
  header("QUERY 2 — Embedding model dimensionality");

  const q2 = "How do embedding model dimensionality and L2 normalisation affect vector search quality?";
  console.log(`  Query: "${q2}"\n`);

  const childHits2 = await searchChildren(q2, store, mockEmbed, 4);
  console.log("  Step 1 — Top-4 children from vector search:");
  printChildScores(childHits2);

  const results2 = await retrieve(q2, store, mockEmbed, 4);
  console.log("\n  Step 2 — Fetch parents, deduplicate:");
  printResults(results2);

  // -----------------------------------------------------------------------
  // QUERY 3 — cross-cutting query that spans multiple parents
  // -----------------------------------------------------------------------
  header("QUERY 3 — Cross-cutting query (spans two parent sections)");

  const q3 = "What strategies reduce index size: chunking overlap, IVFFlat nlist, or dimensionality truncation?";
  console.log(`  Query: "${q3}"\n`);

  const childHits3 = await searchChildren(q3, store, mockEmbed, 6);
  console.log("  Step 1 — Top-6 children from vector search:");
  printChildScores(childHits3);

  const results3 = await retrieve(q3, store, mockEmbed, 6);
  console.log("\n  Step 2 — Fetch parents, deduplicate:");
  console.log("  (Multiple children from different parents → multiple full sections returned)");
  printResults(results3);

  // -----------------------------------------------------------------------
  // DEDUPLICATION DEMO
  // -----------------------------------------------------------------------
  header("DEDUPLICATION — why it matters");

  console.log("  Query 1 retrieved these children:");
  childHits1.forEach((cs) => {
    console.log(`    childId=${cs.child.id.padEnd(34)} → parentId=${cs.child.parentId}`);
  });
  console.log(`\n  Without dedup: ${childHits1.length} LLM context slots used`);
  console.log(`  With    dedup: ${results1.length} LLM context slot${results1.length === 1 ? "" : "s"} used (unique parents)`);
  console.log("\n  → The LLM receives one full parent section per unique parent,");
  console.log("    not a fragmented list of overlapping child excerpts.");

  console.log(`\n${HR}`);
  console.log("Done. Replace mockEmbed with a real provider (Voyage AI, Cohere, OpenAI, …)");
  console.log("and replace the in-memory store with pgvector + a parent key-value table.");
  console.log(HR);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
