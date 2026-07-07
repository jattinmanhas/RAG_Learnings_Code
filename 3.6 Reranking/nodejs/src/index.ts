/**
 * index.ts — runnable demo for Reranking.
 *
 *   npm start
 *
 * No API key required. All models are lexical mocks:
 *   - mockEmbed        : 6-dim concept vector (bi-encoder stand-in)
 *   - mockCrossEncoder : joint query+doc term scoring (cross-encoder stand-in)
 *   - mockTokenEmbed   : one concept vector per word (ColBERT stand-in)
 *   - mockLlmJudge     : rubric grader 0–10 (LLM judge stand-in)
 * In production, swap these for Cohere Rerank / Voyage rerank-2 / a BGE
 * cross-encoder / a Claude judge. The reranker.ts logic doesn't change.
 *
 * The demo walks the strategies in order:
 *   1. Stage-1 bi-encoder baseline  — fast but coarse; order is imperfect.
 *   2. Cross-encoder rerank         — fixes the order.
 *   3. Late interaction (MaxSim)    — the middle ground.
 *   4. LLM rerank                   — pointwise vs pairwise vs listwise.
 *   5. MMR                          — demotes near-duplicates.
 *   6. RRF                          — fuses dense + keyword ranked lists.
 *   7. Business rules               — recency + source authority.
 */

import {
  Chunk,
  Scored,
  biEncoderRetrieve,
  businessRuleRerank,
  crossEncoderRerank,
  lateInteractionRerank,
  llmListwiseRerank,
  llmPairwiseRerank,
  llmPointwiseRerank,
  mmrRerank,
  rrfRerank,
} from "./reranker";

// ---------------------------------------------------------------------------
// SAMPLE CORPUS
// ---------------------------------------------------------------------------
// Same knowledge-base theme as 3.5 (vector databases), built to expose the
// failure modes reranking fixes:
//   - "hnsw-tuning" is the true best answer for the demo query.
//   - "hnsw-rebuild" shares the query's keywords (HNSW, faster, recall) but
//     answers a DIFFERENT question — a classic bi-encoder trap.
//   - "hnsw-tuning-forum" near-duplicates the best answer (MMR fodder), and
//     is old + low-authority (business-rule fodder).

const corpus: Chunk[] = [
  {
    id: "hnsw-tuning",
    text: "To speed up HNSW queries without losing recall, tune ef_search: lower values are faster, higher values improve recall. Set M and ef_construction at build time for a better speed/recall frontier.",
    metadata: { source: "docs", ageDays: "30" },
  },
  {
    id: "hnsw-overview",
    text: "HNSW is a graph-based approximate nearest neighbour index. It builds a multi-layer proximity graph and offers high recall with real-time inserts.",
    metadata: { source: "docs", ageDays: "60" },
  },
  {
    id: "hnsw-tuning-forum",
    text: "Forum answer: basically you make HNSW faster by dropping ef_search, and you get recall back by raising it again. Also worth tweaking M and ef_construction when you build the index.",
    metadata: { source: "forum", ageDays: "700" },
  },
  {
    id: "ivfflat-tuning",
    text: "For IVFFlat, raising the probes parameter improves recall but slows queries; lowering it speeds them up. It is a different tradeoff curve than graph indexes.",
    metadata: { source: "docs", ageDays: "45" },
  },
  {
    id: "quantization",
    text: "Product quantization compresses vectors into compact codes, making search faster and indexes smaller, at the cost of a small recall loss unless you rerank with full-precision vectors.",
    metadata: { source: "blog", ageDays: "200" },
  },
  {
    id: "hnsw-rebuild",
    text: "Changelog: rebuilding an HNSW index is now 3x faster thanks to parallel graph construction. Recall is unaffected because the rebuild produces an identical graph.",
    metadata: { source: "changelog", ageDays: "10" },
  },
  {
    id: "normalization",
    text: "Embeddings should be L2-normalized before cosine similarity so that vector length does not distort semantic distance.",
    metadata: { source: "docs", ageDays: "90" },
  },
];

const QUERY = "How can I make HNSW vector search faster without losing recall?";

// ---------------------------------------------------------------------------
// MOCK MODELS
// ---------------------------------------------------------------------------

// Concept dimensions for the bi-encoder mock. Coarse on purpose — a real
// bi-encoder is also coarse: it compresses a whole passage into one vector.
const CONCEPTS: Record<string, string[]> = {
  hnsw: ["hnsw", "graph", "layer", "m,", "ef_search", "ef_construction"],
  speed: ["faster", "fast", "speed", "latency", "slows", "quick"],
  recall: ["recall", "accuracy", "quality"],
  index: ["index", "indexes", "ivfflat", "probes", "build", "rebuild"],
  vectors: ["vector", "vectors", "embedding", "embeddings", "quantization", "cosine"],
  ops: ["changelog", "forum", "normalized", "parallel"],
};

const tokenize = (text: string): string[] =>
  text.toLowerCase().split(/[^a-z0-9_,]+/).filter(Boolean);

function mockEmbed(text: string): number[] {
  const tokens = tokenize(text);
  return Object.values(CONCEPTS).map(
    (words) => tokens.filter((t) => words.includes(t)).length / tokens.length
  );
}

// Cross-encoder mock: because it sees query and doc TOGETHER, it can reward
// docs that contain the query's rare terms near each other, and penalize
// docs that merely mention the topic. Real cross-encoders learn this from
// data; the point here is the interface and the effect on ordering.
const STOPWORDS = new Set(["how", "can", "i", "make", "the", "a", "an", "without", "to", "of", "is", "it", "and", "you", "by"]);

function mockCrossEncoder(query: string, doc: string): number {
  const qTerms = tokenize(query).filter((t) => !STOPWORDS.has(t));
  const dTokens = tokenize(doc);
  const dSet = new Set(dTokens);

  // Term coverage: what fraction of query terms the doc actually contains.
  const hits = qTerms.filter((t) => dSet.has(t));
  let score = hits.length / qTerms.length;

  // Joint-attention stand-in: reward actionable answers ("tune X", "lower/
  // raise Y") over descriptions. A real cross-encoder picks this up from
  // relevance labels; we approximate with instruction verbs.
  const actionable = ["tune", "set", "raising", "raise", "lowering", "lower", "dropping", "increase"];
  if (dTokens.some((t) => actionable.includes(t))) score += 0.35;

  // Penalize keyword-only matches: doc mentions the terms but is about a
  // different subject (our changelog trap talks about REBUILDING, not querying).
  if (dSet.has("rebuild") || dSet.has("rebuilding")) score -= 0.4;

  return score;
}

// ColBERT mock: one vector per token instead of one per document. Each token
// vector = concept dims (soft, shared across synonyms) + an identity dim from
// a small hash (so an EXACT token match scores higher than a same-concept
// match) — a crude imitation of contextualized token embeddings.
function mockTokenEmbed(text: string): number[][] {
  const ID_DIMS = 8;
  return tokenize(text)
    .filter((t) => !STOPWORDS.has(t))
    .map((token) => {
      const vec = Object.values(CONCEPTS).map((words) =>
        words.includes(token) ? 0.6 : 0
      );
      let hash = 0;
      for (const ch of token) hash = (hash * 31 + ch.charCodeAt(0)) % ID_DIMS;
      const id = new Array(ID_DIMS).fill(0);
      id[hash] = 1;
      return [...vec, ...id];
    });
}

// LLM judge mock: a deterministic "rubric" built on the cross-encoder score,
// snapped to a 0–10 integer — mimicking a temperature-0 grading prompt.
function mockLlmJudge(query: string, doc: string): number {
  return Math.max(0, Math.min(10, Math.round(mockCrossEncoder(query, doc) * 7)));
}

const mockLlmDuel = (query: string, a: string, b: string): "a" | "b" =>
  mockCrossEncoder(query, a) >= mockCrossEncoder(query, b) ? "a" : "b";

// Listwise mock: the model "reads" all passages in one prompt and emits an
// order. RankGPT would return "[1] > [3] > …"; we return the indices.
const mockLlmListRank = (query: string, docs: string[]): number[] =>
  docs
    .map((doc, i) => ({ i, s: mockCrossEncoder(query, doc) }))
    .sort((x, y) => y.s - x.s)
    .map(({ i }) => i);

// Keyword retriever (BM25-lite) for the RRF demo — rank by raw query-term hits.
function keywordRetrieve(query: string, chunks: Chunk[], topK: number): Scored[] {
  const qTerms = tokenize(query).filter((t) => !STOPWORDS.has(t));
  return chunks
    .map((chunk) => {
      const dTokens = tokenize(chunk.text);
      const score = qTerms.reduce(
        (acc, t) => acc + dTokens.filter((d) => d === t).length,
        0
      );
      return { chunk, score };
    })
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ---------------------------------------------------------------------------
// DEMO
// ---------------------------------------------------------------------------

function printRanked(title: string, ranked: Scored[]): void {
  console.log(`\n${title}`);
  ranked.forEach(({ chunk, score }, i) =>
    console.log(`  ${i + 1}. [${score.toFixed(3)}] ${chunk.id} (${chunk.metadata.source})`)
  );
}

console.log(`Query: "${QUERY}"`);

// 1. Stage 1 — bi-encoder over the whole corpus. Watch the shortlist: the
// changelog trap makes the cut and off-topic quantization outranks docs that
// actually answer the question, because one-vector-per-doc similarity can't
// tell "mentions HNSW + faster + recall" apart from "tells you how".
const candidates = biEncoderRetrieve(QUERY, corpus, mockEmbed, 6);
printRanked("1. STAGE 1 — bi-encoder retrieval (coarse, order imperfect)", candidates);

// 2. Cross-encoder rerank — reads each (query, doc) pair jointly; the trap
// drops, the actionable answer rises to #1.
printRanked(
  "2. CROSS-ENCODER rerank (production standard)",
  crossEncoderRerank(QUERY, candidates, mockCrossEncoder, 5)
);

// 3. Late interaction — token-level MaxSim. Better than the bi-encoder
// because each query token matches its best doc token independently.
printRanked(
  "3. LATE INTERACTION rerank (ColBERT-style MaxSim)",
  lateInteractionRerank(QUERY, candidates, mockTokenEmbed, 5)
);

// 4. LLM reranking, three flavors over the same candidates.
printRanked(
  "4a. LLM POINTWISE rerank (grade each doc 0–10)",
  llmPointwiseRerank(QUERY, candidates, mockLlmJudge, 5)
);
printRanked(
  "4b. LLM PAIRWISE rerank (round-robin duels, score = wins)",
  llmPairwiseRerank(QUERY, candidates, mockLlmDuel, 5)
);
printRanked(
  "4c. LLM LISTWISE rerank (RankGPT-style, one call ranks all)",
  llmListwiseRerank(QUERY, candidates, mockLlmListRank, 5)
);

// 5. MMR — note the forum near-duplicate of the top answer gets pushed down
// in favor of docs that add NEW information (quantization, ivfflat).
printRanked(
  "5. MMR rerank (λ=0.6 — relevance + diversity, duplicate demoted)",
  mmrRerank(QUERY, candidates, mockEmbed, 0.6, 4)
);

// 6. RRF — fuse the dense list with a keyword list using ranks only.
const denseList = biEncoderRetrieve(QUERY, corpus, mockEmbed, 6);
const keywordList = keywordRetrieve(QUERY, corpus, 6);
printRanked("6a. dense list (for fusion)", denseList);
printRanked("6b. keyword list (for fusion)", keywordList);
printRanked("6c. RRF fusion (k=60)", rrfRerank([denseList, keywordList], 60, 5));

// 7. Business rules — applied AFTER the cross-encoder. The 700-day-old forum
// paraphrase sinks below fresher, authoritative docs even though its pure
// relevance score was nearly identical.
const relevanceRanked = crossEncoderRerank(QUERY, candidates, mockCrossEncoder, 5);
printRanked(
  "7. BUSINESS-RULE rerank (recency half-life 365d, docs boosted, forum damped)",
  businessRuleRerank(
    relevanceRanked,
    { sourceWeights: { docs: 1.2, changelog: 1.0, blog: 0.9, forum: 0.6 }, halfLifeDays: 365 },
    5
  )
);

console.log(
  "\nProduction recipe: bi-encoder/hybrid top-50 → cross-encoder top-10 → " +
    "MMR/business rules → top-3..5 into the context window."
);
