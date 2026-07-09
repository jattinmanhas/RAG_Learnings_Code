// index.ts — runnable demo for Retrieval Confidence.
//
//   npm start
//
// No API key required. Mocks:
//   - mockEmbed  : 6-dim concept vector (embedding stand-in)
//   - mock judges: lexical overlap heuristics (LLM stand-ins)
//
// Three queries against a vector-DB knowledge base:
//   Q1 answerable   — corpus covers it well
//   Q2 ambiguous    — corpus only partially covers it
//   Q3 unanswerable — off-corpus (cooking); retrieval still "returns" chunks!
//
// For each family of signals we print what it reads on all three queries,
// then run the Complete Confidence Pipeline end to end.

import {
  Hit,
  similarityFloor,
  top1Floor,
  countAboveThreshold,
  calibrateThreshold,
  top1Top2Gap,
  relativeDropoff,
  softmaxEntropy,
  zScoreVsBackground,
  crossRetrieverAgreement,
  answerabilityCheck,
  gradePassages,
  groundednessCheck,
  selfConsistency,
  confidencePipeline,
  Answerability,
} from "./confidence";

// ---------------------------------------------------------------------------
// SAMPLE CORPUS — same vector-database theme as 3.5/3.6
// ---------------------------------------------------------------------------

interface Doc {
  id: string;
  text: string;
}

const CORPUS: Doc[] = [
  {
    id: "hnsw-tuning",
    text: "To make HNSW faster without hurting recall, increase efSearch gradually and tune M; measure recall on a held-out query set.",
  },
  {
    id: "hnsw-basics",
    text: "HNSW is a graph-based index for approximate nearest neighbour search with strong recall/latency trade-offs.",
  },
  {
    id: "ivf-index",
    text: "IVF indexes partition vectors into clusters; probe more lists for better recall at higher latency.",
  },
  {
    id: "pgvector-ops",
    text: "pgvector supports cosine, L2 and inner-product operators; build an HNSW index for production workloads.",
  },
  {
    id: "embedding-models",
    text: "Choosing an embedding model: balance dimensionality, cost and retrieval quality; re-embed when you switch models.",
  },
  {
    id: "chunking",
    text: "Chunking strategy affects retrieval: overlapping windows help recall, semantic chunking keeps ideas intact.",
  },
];

// ---------------------------------------------------------------------------
// MOCK EMBEDDINGS — 6 concept dimensions
// [hnsw, index/search, recall/latency, postgres, embeddings, food]
// ---------------------------------------------------------------------------

const CONCEPTS: [RegExp, number][] = [
  [/hnsw|efsearch|\bm\b|graph/i, 0],
  [/index|search|ivf|nearest|probe|lists/i, 1],
  [/recall|latency|faster|speed|trade/i, 2],
  [/pgvector|postgres|operator/i, 3],
  [/embed|model|dimension|chunk/i, 4],
  [/cook|pasta|carbonara|recipe|egg|cheese/i, 5],
];

function mockEmbed(text: string): number[] {
  const v = new Array(6).fill(0);
  for (const word of text.toLowerCase().split(/\W+/)) {
    for (const [re, dim] of CONCEPTS) if (re.test(word)) v[dim] += 1;
  }
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

const cosine = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + x * b[i], 0);

function retrieve(query: string, k = 4): Hit[] {
  const qv = mockEmbed(query);
  return CORPUS.map((d) => ({
    id: d.id,
    text: d.text,
    score: cosine(qv, mockEmbed(d.text)),
  }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/** Keyword (BM25-ish) retrieval mock: rank by term overlap. */
function keywordRetrieve(query: string, k = 4): string[] {
  const qTerms = new Set(query.toLowerCase().split(/\W+/).filter((w) => w.length > 3));
  return CORPUS.map((d) => ({
    id: d.id,
    overlap: d.text
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => qTerms.has(w)).length,
  }))
    .sort((a, b) => b.overlap - a.overlap)
    .filter((d) => d.overlap > 0)
    .slice(0, k)
    .map((d) => d.id);
}

// ---------------------------------------------------------------------------
// MOCK LLM JUDGES — lexical overlap standing in for Claude
// ---------------------------------------------------------------------------

const overlap = (a: string, b: string): number => {
  const ta = new Set(a.toLowerCase().split(/\W+/).filter((w) => w.length > 3));
  const tb = new Set(b.toLowerCase().split(/\W+/).filter((w) => w.length > 3));
  let n = 0;
  for (const t of ta) if (tb.has(t)) n++;
  return ta.size === 0 ? 0 : n / ta.size;
};

const mockAnswerabilityJudge = (
  query: string,
  passages: string[]
): Answerability => {
  const best = Math.max(...passages.map((p) => overlap(query, p)));
  if (best >= 0.4) return "YES";
  if (best >= 0.15) return "PARTIAL";
  return "NO";
};

const mockGrader = (query: string, passage: string): number =>
  Math.round(overlap(query, passage) * 10);

const mockEntails = (claim: string, evidence: string[]): boolean =>
  evidence.some((e) => overlap(claim, e) >= 0.5);

/** Extractive "generator": answers with the top passage, claims = sentences. */
const mockGenerate = (_query: string, hits: Hit[]) => {
  const answer = hits[0].text;
  return { answer, claims: answer.split(/(?<=[.;])\s+/).filter(Boolean) };
};

// ---------------------------------------------------------------------------
// DEMO
// ---------------------------------------------------------------------------

const QUERIES = [
  { label: "Q1 answerable  ", q: "How do I make HNSW search faster without hurting recall?" },
  { label: "Q2 ambiguous   ", q: "Should I re-embed my chunks when switching from IVF to HNSW in Postgres?" },
  { label: "Q3 unanswerable", q: "What is the best recipe for pasta carbonara with eggs and cheese?" },
];

const section = (title: string) => console.log(`\n${"=".repeat(75)}\n${title}\n${"=".repeat(75)}`);

// --- 1. Absolute thresholding ------------------------------------------------
section("1. ABSOLUTE THRESHOLDING (crude, corpus-specific)");

// 1d: calibrate τ from known-irrelevant pairs instead of guessing.
const irrelevantScores = ["carbonara recipe", "football scores", "weather tomorrow"]
  .flatMap((q) => retrieve(q, CORPUS.length).map((h) => h.score));
const tau = Math.max(0.15, calibrateThreshold(irrelevantScores, 0.95));
console.log(`calibrated τ from irrelevant-pair scores (95th pct): ${tau.toFixed(3)}\n`);

for (const { label, q } of QUERIES) {
  const hits = retrieve(q);
  console.log(`${label} top-1=${hits[0].score.toFixed(3)} (${hits[0].id})`);
  console.log(`   1a similarity floor : ${similarityFloor(hits, tau).length}/${hits.length} hits survive τ=${tau.toFixed(2)}`);
  console.log(`   1b top-1 floor      : ${top1Floor(hits, tau) ? "proceed" : "REFUSE"}`);
  console.log(`   1c count ≥2 above τ : ${countAboveThreshold(hits, tau, 2) ? "yes" : "no"}`);
  console.log(`   1e reranker floor   : ${gradePassages(q, hits, mockGrader, 5).length} hits pass LLM-grade ≥5`);
}

// --- 2. Relative / distributional signals ------------------------------------
section("2. RELATIVE / DISTRIBUTIONAL SIGNALS (more robust)");

// 2d: background similarity of random corpus chunks to each query.
for (const { label, q } of QUERIES) {
  const hits = retrieve(q);
  const background = CORPUS.map((d) => cosine(mockEmbed(q), mockEmbed(d.text)));
  const denseIds = hits.map((h) => h.id);
  const kwIds = keywordRetrieve(q);
  console.log(`${label}`);
  console.log(`   2a top1-top2 gap    : ${top1Top2Gap(hits).toFixed(3)}`);
  console.log(`   2b relative dropoff : ${relativeDropoff(hits).toFixed(2)}x`);
  console.log(`   2c softmax entropy  : ${softmaxEntropy(hits).toFixed(3)}  (0=peaked, 1=uniform)`);
  console.log(`   2d z-score vs corpus: ${zScoreVsBackground(hits[0].score, background).toFixed(2)}`);
  console.log(`   2e dense∩keyword    : ${crossRetrieverAgreement(denseIds, kwIds).toFixed(2)} Jaccard`);
}

// --- 3. LLM-based groundedness checks -----------------------------------------
section("3. LLM-BASED GROUNDEDNESS CHECKS (most robust, most expensive)");

for (const { label, q } of QUERIES) {
  const hits = retrieve(q);
  const gate = answerabilityCheck(q, hits, mockAnswerabilityJudge);
  console.log(`${label}`);
  console.log(`   3a answerability    : ${gate}`);
  console.log(`   3b passage grades   : [${hits.map((h) => mockGrader(q, h.text)).join(", ")}] → ${gradePassages(q, hits, mockGrader).length} kept`);
  const { claims } = mockGenerate(q, hits);
  const g = groundednessCheck(claims, hits, mockEntails);
  console.log(`   3c groundedness     : ${g.supported.length}/${claims.length} claims supported`);
  const sc = selfConsistency(() => hits[Math.random() < (gate === "YES" ? 0.95 : 0.4) ? 0 : 1].id, 5);
  console.log(`   3e self-consistency : ${(sc.agreement * 100).toFixed(0)}% of 5 samples agree`);
}

// --- 4. The Complete Confidence Pipeline ---------------------------------------
section("4. THE COMPLETE CONFIDENCE PIPELINE (cheapest-first, early exits)");

const cfg = {
  tauHard: tau,
  gapConfident: 0.15,
  entropyConfident: 0.5,
  entropyHopeless: 0.9,
  groundednessMin: 0.7,
};

for (const { label, q } of QUERIES) {
  const res = confidencePipeline(q, retrieve(q), cfg, {
    answerabilityJudge: mockAnswerabilityJudge,
    generate: mockGenerate,
    entails: mockEntails,
  });
  console.log(`\n${label} "${q}"`);
  for (const line of res.trace) console.log(`   ${line}`);
  console.log(`   → ${res.verdict}  (decided at ${res.decidedAt})`);
  if (res.answer) console.log(`   answer: ${res.answer}`);
}
