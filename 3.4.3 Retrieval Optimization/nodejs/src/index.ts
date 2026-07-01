/**
 * index.ts — runnable demo for Retrieval Optimization.
 *
 *   npm start
 *
 * No API key required. mockEmbed turns text into a 6-dimensional concept vector
 * via keyword scoring — enough to show dense retrieval, and to contrast it with
 * BM25 keyword retrieval, without a network call. Replace mockEmbed with a real
 * provider (Voyage AI, Cohere, OpenAI) in production.
 *
 * The demo walks the four levers in order:
 *   1. Top-K retrieval    — dense baseline, varying K.
 *   2. Metadata filtering — same query, restricted by structured fields.
 *   3. Hybrid retrieval   — dense vs sparse vs fused on an exact-token query.
 *   4. Retrieval tuning   — sweeping alpha, RRF vs weighted, thresholds.
 */

import {
  Chunk,
  EmbedFn,
  FusionMode,
  Scored,
  Store,
  and,
  defaultOptions,
  fieldEquals,
  fieldIn,
  matchAll,
} from "./retriever";

// ---------------------------------------------------------------------------
// SAMPLE CORPUS
// ---------------------------------------------------------------------------
// A small knowledge base about vector databases. Each chunk carries metadata
// you'd realistically filter on: source, language, access level. Note the exact
// tokens ("HNSW", "E1401", "pgvector 0.7") — those are where dense-only
// retrieval struggles and BM25 shines.

const corpus: Chunk[] = [
  {
    id: "hnsw-overview",
    text: "HNSW is a graph-based approximate nearest neighbour index. It builds a multi-layer proximity graph and offers high recall with real-time inserts.",
    metadata: { source: "docs", language: "en", access: "public" },
  },
  {
    id: "ivfflat-overview",
    text: "IVFFlat partitions the vector space into Voronoi cells with k-means. It uses less memory than HNSW but must be rebuilt when the cluster count changes.",
    metadata: { source: "docs", language: "en", access: "public" },
  },
  {
    id: "pgvector-release",
    text: "pgvector 0.7 adds halfvec and sparse vector support, improving memory usage for large embedding tables inside Postgres.",
    metadata: { source: "changelog", language: "en", access: "public" },
  },
  {
    id: "error-e1401",
    text: "Troubleshooting error code E1401: the embedding dimension of the query does not match the index dimension. Re-embed with the correct model.",
    metadata: { source: "support", language: "en", access: "internal" },
  },
  {
    id: "recall-tuning",
    text: "To raise recall, increase the number of candidates retrieved before re-ranking, or widen the search parameter so the graph traversal visits more nodes.",
    metadata: { source: "docs", language: "en", access: "public" },
  },
  {
    id: "memoria-vectorial",
    text: "La memoria vectorial almacena embeddings densos para la busqueda semantica. HNSW ofrece alta exhaustividad con inserciones en tiempo real.",
    metadata: { source: "docs", language: "es", access: "public" },
  },
];

// ---------------------------------------------------------------------------
// MOCK EMBEDDING FUNCTION
// ---------------------------------------------------------------------------
// Maps text to a 6-dimensional concept vector by counting keyword matches per
// semantic dimension. Deliberately blind to exact tokens like "E1401" — that
// blindness is precisely why the hybrid demo needs BM25.
//
// Dimension legend:
//   [0] graph-index / HNSW concepts
//   [1] partition / IVFFlat / clustering concepts
//   [2] memory / storage concepts
//   [3] recall / tuning concepts
//   [4] error / troubleshooting concepts
//   [5] Postgres / pgvector concepts

const conceptKeywords: string[][] = [
  ["hnsw", "graph", "proximity", "layer", "navigable", "node", "traversal", "exhaustividad", "vectorial"],
  ["ivfflat", "ivf", "voronoi", "cluster", "kmeans", "k-means", "partition", "centroid"],
  ["memory", "storage", "halfvec", "usage", "memoria", "densos"],
  ["recall", "candidate", "rerank", "re-ranking", "tuning", "search", "widen", "raise"],
  ["error", "troubleshoot", "troubleshooting", "mismatch", "dimension", "match"],
  ["pgvector", "postgres", "sparse", "table", "embedding"],
];

function countOccurrences(haystack: string, needle: string): number {
  if (needle.length === 0) return 0;
  let count = 0;
  let idx = haystack.indexOf(needle);
  while (idx !== -1) {
    count++;
    idx = haystack.indexOf(needle, idx + needle.length);
  }
  return count;
}

const mockEmbed: EmbedFn = async (text: string): Promise<number[]> => {
  const lower = text.toLowerCase();
  const vec = conceptKeywords.map((kws) =>
    kws.reduce((sum, kw) => sum + countOccurrences(lower, kw), 0)
  );
  return l2Normalize(vec);
};

function l2Normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return norm === 0 ? v : v.map((x) => x / norm);
}

// ---------------------------------------------------------------------------
// DISPLAY HELPERS
// ---------------------------------------------------------------------------

const HR = "============================================================================";

function header(title: string): void {
  console.log(`\n${HR}\n${title}\n${HR}`);
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n) + "…";
}

function printResults(results: Scored[], showComponents: boolean): void {
  if (results.length === 0) {
    console.log("  (no results)");
    return;
  }
  results.forEach((r, i) => {
    let line = `  [${i + 1}] ${r.chunk.id.padEnd(18)} score=${r.score.toFixed(4)}`;
    if (showComponents) {
      line += `  (dense=${r.denseScore.toFixed(3)}#${r.denseRank}  sparse=${r.sparseScore.toFixed(3)}#${r.sparseRank})`;
    }
    console.log(line);
    console.log(`        "${truncate(r.chunk.text, 78)}"`);
  });
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const store = new Store(corpus);

  // =======================================================================
  // TOPIC 1 — TOP-K RETRIEVAL
  // =======================================================================
  header("TOPIC 1 — Top-K Retrieval (dense only)");

  const q1 = "How does HNSW indexing work and when should I use it?";
  console.log(`  Query: "${q1}"`);

  for (const k of [2, 4]) {
    const res = await store.retrieve(q1, mockEmbed, { hybrid: false, topK: k });
    console.log(`\n  TopK = ${k}:`);
    printResults(res, false);
  }
  console.log("\n  → K trades recall for precision/latency. Too small and you miss a");
  console.log("    relevant chunk; too large and you pad the prompt with noise.");

  // =======================================================================
  // TOPIC 2 — METADATA FILTERING
  // =======================================================================
  header("TOPIC 2 — Metadata Filtering");

  const q2 = "vector memory for semantic search";
  console.log(`  Query: "${q2}"`);

  console.log("\n  No filter (all languages, all access levels):");
  printResults(await store.retrieve(q2, mockEmbed, { hybrid: false }), false);

  console.log('\n  Filter: language == "en"  (drop the Spanish chunk):');
  printResults(
    await store.retrieve(q2, mockEmbed, { hybrid: false, filter: fieldEquals("language", "en") }),
    false
  );

  console.log('\n  Filter: language == "en" AND access == "public"  (security filter):');
  printResults(
    await store.retrieve(q2, mockEmbed, {
      hybrid: false,
      filter: and(fieldEquals("language", "en"), fieldIn("access", "public")),
    }),
    false
  );
  console.log("\n  → Filtering runs BEFORE scoring. The 'internal' error-E1401 chunk can");
  console.log("    never leak to a public user, regardless of how well it matches.");

  // =======================================================================
  // TOPIC 3 — HYBRID RETRIEVAL
  // =======================================================================
  header("TOPIC 3 — Hybrid Retrieval (dense + BM25)");

  // An exact-token lookup: the mock embedder has never seen "E1401", so it
  // can't tell the chunks apart — but BM25 nails it. Hybrid fusion recovers the
  // exact match the vectors miss.
  const q3 = "look up the E1401 issue";
  console.log(`  Query: "${q3}"`);
  console.log("  (note: the mock embedder has no concept for the exact token 'E1401')");

  console.log("\n  Dense only — can't distinguish, the exact match is not surfaced:");
  printResults(await store.retrieve(q3, mockEmbed, { hybrid: false, filter: matchAll, topK: 3 }), true);

  console.log("\n  Hybrid (RRF fusion) — BM25 surfaces the exact match, fused to the top:");
  printResults(
    await store.retrieve(q3, mockEmbed, {
      filter: matchAll,
      topK: 3,
      fusion: FusionMode.RRF,
    }),
    true
  );
  console.log("\n  → Dense gives semantic recall; BM25 gives exact-token precision.");
  console.log("    RRF fuses the two rankings without needing to normalise scores.");

  // =======================================================================
  // TOPIC 4 — RETRIEVAL TUNING
  // =======================================================================
  header("TOPIC 4 — Retrieval Tuning (sweeping the knobs)");

  const q4 = "how to raise recall with HNSW";
  console.log(`  Query: "${q4}"`);

  console.log("\n  (a) Weighted fusion, sweeping alpha (dense weight):");
  for (const alpha of [0.0, 0.5, 1.0]) {
    const res = await store.retrieve(q4, mockEmbed, {
      filter: matchAll,
      topK: 3,
      fusion: FusionMode.Weighted,
      alpha,
    });
    const label =
      alpha === 0.0 ? "alpha=0.0 (pure BM25)" : alpha === 1.0 ? "alpha=1.0 (pure dense)" : "alpha=0.5";
    console.log(`\n    ${label}:`);
    printResults(res, true);
  }

  console.log("\n  (b) RRF constant — smaller rrfK sharpens top-rank dominance:");
  for (const rrfK of [5, 60]) {
    const res = await store.retrieve(q4, mockEmbed, {
      filter: matchAll,
      topK: 3,
      fusion: FusionMode.RRF,
      rrfK,
    });
    console.log(`\n    rrfK=${rrfK}:`);
    printResults(res, true);
  }

  console.log("\n  (c) Score threshold — drop weak matches (abstention signal):");
  const threshold = 0.0322;
  const res = await store.retrieve(q4, mockEmbed, {
    filter: matchAll,
    topK: 5,
    fusion: FusionMode.RRF,
    threshold,
  });
  console.log(`    Threshold=${threshold} keeps ${res.length} of up to 5 (weak matches dropped):`);
  printResults(res, true);

  console.log(`\n${HR}`);
  console.log("Done. In production:");
  console.log("  • Replace mockEmbed with a real provider (Voyage AI, Cohere, OpenAI).");
  console.log("  • Replace the in-memory dense search with a pgvector ORDER BY <=> query.");
  console.log("  • Replace BM25 with Postgres full-text (tsvector) or Elasticsearch.");
  console.log("  • Push the metadata filter into the SQL WHERE clause so the index does it.");
  console.log("  • Tune topK / alpha / rrfK / threshold against a labelled eval set (3.10).");
  console.log(HR);
  // defaultOptions is exported for callers who want the production baseline.
  void defaultOptions;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
