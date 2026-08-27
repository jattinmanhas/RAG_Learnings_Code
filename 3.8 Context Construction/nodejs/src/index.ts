// index.ts — runnable demo for Context Construction.
//
//   npm start
//
// No API key required. Mocks:
//   - countTokens : chars/4 (stand-in for tiktoken / count-tokens API)
//   - summarize   : first sentence + query terms (stand-in for a Haiku call)
//
// A retrieval result for one query is carried through every stage: packing
// under a tight budget, the four orderings, the compression cascade,
// citation-ready formatting, and finally the complete pipeline end to end.

import {
  Chunk,
  evidenceBudget,
  packGreedy,
  packByDensity,
  dedupe,
  capPerSource,
  packTiered,
  totalTokens,
  orderByRelevance,
  orderBookend,
  orderByDocument,
  orderGroupedBookend,
  extractiveCompress,
  truncateToTokens,
  pruneRedundantSentences,
  compressionCascade,
  formatForCitation,
  citationInstructions,
  resolveCitations,
  queryAdjacentLayout,
  buildContext,
} from "./context";

// ---------------------------------------------------------------------------
// SAMPLE RETRIEVAL RESULT — same vector-database theme as 3.5–3.7, but now
// every chunk carries the provenance a citation needs (source + loc + ordinal)
// and the list contains the messes real retrieval produces: a near-duplicate,
// one very long chunk, and four hits from the same file.
// ---------------------------------------------------------------------------

const HITS: Chunk[] = [
  {
    id: "c1",
    score: 0.91,
    source: "docs/hnsw-tuning.md",
    loc: "L12-L28",
    ordinal: 2,
    text: "To make HNSW search faster without hurting recall, raise efSearch gradually and re-measure recall on a held-out query set. efSearch is a pure latency/recall dial at query time; M and efConstruction are build-time and require a reindex.",
  },
  {
    id: "c2",
    score: 0.88,
    source: "docs/hnsw-tuning.md",
    loc: "L30-L44",
    ordinal: 3,
    text: "A practical tuning loop: fix M, sweep efSearch over 40, 80, 160, 320 and plot recall@10 against p95 latency. Stop at the knee of the curve. Raising M helps recall on high-dimensional data but costs memory and build time.",
  },
  {
    id: "c3",
    score: 0.86,
    source: "docs/hnsw-tuning.md",
    loc: "L12-L27",
    ordinal: 2,
    // NEAR-DUPLICATE of c1 — overlapping-window chunking produced both.
    text: "To make HNSW search faster without hurting recall, raise efSearch gradually and re-measure recall on a held-out query set. efSearch is a pure latency/recall dial at query time; M and efConstruction are build-time settings.",
  },
  {
    id: "c4",
    score: 0.72,
    source: "docs/hnsw-basics.md",
    loc: "L1-L9",
    ordinal: 1,
    text: "HNSW is a graph-based index for approximate nearest neighbour search, built from a hierarchy of navigable small-world graphs.",
  },
  {
    id: "c5",
    score: 0.69,
    source: "docs/pgvector.md",
    loc: "L60-L120",
    ordinal: 5,
    // The long one — high value per chunk, terrible value per token.
    text: "pgvector supports cosine distance, L2 distance and inner product operators, each with its own operator class. Index selection follows the operator you query with: vector_cosine_ops for cosine, vector_l2_ops for Euclidean, vector_ip_ops for inner product. Building an HNSW index in pgvector takes maintenance_work_mem into account, so raise it before a large build or the build will spill to disk and take hours. The ef_search parameter is set per session with SET hnsw.ef_search, which means an application pool must set it on every connection or fall back to the default of 40. IVFFlat remains available and builds far faster, but its recall degrades as the table grows unless you reindex after significant inserts. For production workloads on tables above a million rows, HNSW is the default recommendation despite the longer build.",
  },
  {
    id: "c6",
    score: 0.64,
    source: "docs/hnsw-tuning.md",
    loc: "L46-L58",
    ordinal: 4,
    text: "Recall is measured against exact brute-force search on the same query set. Never tune efSearch against production traffic without a ground-truth set; you will optimise latency and silently lose answers.",
  },
  {
    id: "c7",
    score: 0.58,
    source: "docs/ivf.md",
    loc: "L1-L14",
    ordinal: 1,
    text: "IVF indexes partition vectors into clusters and probe a subset at query time. Probing more lists improves recall at higher latency. Recall is measured against exact brute-force search on the same query set.",
  },
  {
    id: "c8",
    score: 0.41,
    source: "docs/embeddings.md",
    loc: "L20-L31",
    ordinal: 2,
    text: "Choosing an embedding model means balancing dimensionality, cost and retrieval quality. Switching models requires re-embedding the whole corpus and re-calibrating every threshold you tuned.",
  },
];

const QUERY = "How do I make HNSW search faster without hurting recall?";

// ---------------------------------------------------------------------------
// MOCKS — swap for a real tokenizer and a real LLM
// ---------------------------------------------------------------------------

/** ~4 characters per token is the standard English rule of thumb. Real systems
 *  must use the actual tokenizer: guessing low truncates, guessing high wastes
 *  budget on every single request. */
const countTokens = (text: string): number => Math.ceil(text.length / 4);

/** Stand-in for "summarise this passage with respect to the query" (Haiku). */
const summarize = (text: string, query: string): string => {
  const q = new Set(query.toLowerCase().split(/\W+/).filter(Boolean));
  const first = text.split(/(?<=[.!?;])\s+/)[0] ?? text;
  const keywords = [...new Set(text.toLowerCase().split(/\W+/))]
    .filter((w) => w.length > 4 && q.has(w))
    .slice(0, 4);
  return `${first.slice(0, 90).trim()}… (key: ${keywords.join(", ") || "n/a"})`;
};

const section = (title: string) =>
  console.log(`\n${"=".repeat(78)}\n${title}\n${"=".repeat(78)}`);

console.log(`query: "${QUERY}"`);
console.log(`retrieved: ${HITS.length} chunks, ${totalTokens(HITS, countTokens)} tokens raw`);

// --- 1. Context Packing -------------------------------------------------------
section("1. CONTEXT PACKING — fitting maximum relevant content in the budget");

const budget = evidenceBudget({
  contextWindow: 1000, // deliberately tiny so the trade-offs are visible
  systemTokens: 120,
  queryTokens: countTokens(QUERY),
  answerReserve: 400,
});
console.log(`1a evidence budget      : ${budget} tok (window 1000 − system 120 − query − answer 400 − margin)\n`);

const show = (label: string, cs: Chunk[]) =>
  console.log(
    `${label}: ${totalTokens(cs, countTokens).toString().padStart(3)} tok  [${cs.map((c) => c.id).join(", ")}]`
  );

show("1b greedy (by score)   ", packGreedy(HITS, budget, countTokens));
show("1c density (score/tok) ", packByDensity(HITS, budget, countTokens));
show("1d after dedupe        ", dedupe(HITS, 0.8));
show("1e cap 2 per source    ", capPerSource(HITS, 2));
show(
  "1f tiered (2 full, rest summarised)",
  packTiered(HITS, budget, countTokens, 2, (c) => summarize(c.text, QUERY))
);
console.log(`\n   note: greedy skips c5 (the 209-tok pgvector chunk) and fits c6 + c7 in its place —`);
console.log(`   that single 'continue' instead of 'break' is worth two extra chunks of evidence.`);
console.log(`   dedupe (1d) is free budget: c3 is a near-copy of c1 and buys ~56 tok back.`);

// --- 2. Context Ordering ------------------------------------------------------
section("2. CONTEXT ORDERING — highest relevance at top and bottom (not middle)");

const packed = packGreedy(dedupe(HITS, 0.8), budget, countTokens);
const fmtOrder = (cs: Chunk[]) =>
  cs.map((c) => `${c.id}(${c.score.toFixed(2)})`).join(" → ");

console.log(`2a relevance desc  : ${fmtOrder(orderByRelevance(packed))}`);
console.log(`                     ↑ strongest evidence in the head, WEAKEST in the strong tail slot`);
console.log(`2b bookend (V)     : ${fmtOrder(orderBookend(packed))}`);
console.log(`                     ↑ ranks 1 & 2 at both attention peaks, weakest buried mid-context`);
console.log(`2c document order  : ${fmtOrder(orderByDocument(packed))}`);
console.log(`                     ↑ contiguous runs per file, files ranked by their best chunk`);
console.log(`2d grouped bookend : ${fmtOrder(orderGroupedBookend(packed))}`);
console.log(`                     ↑ coherent per-file runs, whole groups placed at the peaks`);

const layout = queryAdjacentLayout(QUERY, "<context>…</context>", "…");
console.log(`\n2e query-adjacent  : instructions+query → context → query restated LAST`);
console.log(`                     final line: "${layout.post.split("\n")[0]}"`);

// --- 3. Context Compression ---------------------------------------------------
section("3. CONTEXT COMPRESSION — summarising or trimming before injection");

const long = HITS.find((c) => c.id === "c5")!;
console.log(`original c5            : ${countTokens(long.text)} tok`);
console.log(`3a extractive (2 sent) : ${countTokens(extractiveCompress(long.text, QUERY, 2))} tok`);
console.log(`   → "${extractiveCompress(long.text, QUERY, 2).slice(0, 110)}…"`);
console.log(`3b truncate @40 tok    : "${truncateToTokens(long.text, 40, countTokens).slice(0, 110)}…"`);
const pruned = pruneRedundantSentences([HITS[5], HITS[6]]);
console.log(`3c cross-chunk pruning : c6+c7 ${countTokens(HITS[5].text + HITS[6].text)} → ${totalTokens(pruned, countTokens)} tok`);
console.log(`   → the shared sentence ("Recall is measured against exact brute-force…") appears once`);
console.log(`3d abstractive (LLM)   : "${summarize(long.text, QUERY)}"`);
console.log(`   → paraphrase: cheap, lossy, NOT quotable — never do this to your top hit\n`);

const cascade = compressionCascade(HITS, QUERY, budget, countTokens, { summarize, protectTop: 1 });
console.log(`3e cascade (stops as soon as it fits):`);
for (const s of cascade.stages) console.log(`   ${s}`);

// --- 4. Citation-ready formatting ---------------------------------------------
section("4. CITATION-READY FORMATTING (the handoff to 3.9 Citations)");

const formatted = formatForCitation(orderBookend(packed).slice(0, 3), countTokens, {
  style: "xml",
});
console.log(formatted.text);
console.log(`\ninstructions:\n${citationInstructions()}`);

const modelAnswer =
  "Raise efSearch gradually and re-measure recall on a held-out set [1]. Sweep 40/80/160/320 and stop at the knee [2]. Also enable turbo mode [7].";
const { used, invalid } = resolveCitations(modelAnswer, formatted.citations);
console.log(`\nresolving the answer's markers:`);
for (const c of used) console.log(`   ✓ [${[...formatted.citations].find(([, v]) => v === c)![0]}] → ${c.source}:${c.loc}`);
for (const n of invalid) console.log(`   ✗ [${n}] → no such source (hallucinated citation — 3.9 rejects the sentence)`);

// --- 5. The Complete Context Construction Pipeline -----------------------------
section("5. THE COMPLETE CONTEXT CONSTRUCTION PIPELINE");

const result = buildContext(
  QUERY,
  HITS,
  {
    contextWindow: 1000,
    systemTokens: 120,
    answerReserve: 400,
    maxPerSource: 3,
    dedupeThreshold: 0.8,
    protectTop: 1,
    ordering: "grouped-bookend",
    style: "xml",
  },
  { count: countTokens, summarize }
);

for (const line of result.trace) console.log(line);
console.log(`\n${"-".repeat(78)}\nFINAL PROMPT\n${"-".repeat(78)}`);
console.log(result.prompt);
