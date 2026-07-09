// confidence.ts — Retrieval confidence signals for RAG.
//
// Three families, cheapest → most expensive:
//   1. Absolute thresholding        (crude, corpus-specific)
//   2. Relative/distributional      (more robust, still free)
//   3. LLM-based groundedness       (most robust, costs LLM calls)
// plus the Complete Confidence Pipeline that composes them cheapest-first.
//
// Everything here is model-agnostic: scores come in as plain numbers, LLM
// judges come in as injected functions, so the mocks in index.ts can be
// swapped for real embedding / Claude calls without touching this file.

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

/** A retrieved chunk with its similarity score (higher = more similar). */
export interface Hit {
  id: string;
  text: string;
  score: number;
}

export type Verdict = "ANSWER" | "HEDGE" | "REFUSE";

// ===========================================================================
// 1. ABSOLUTE THRESHOLDING (crude, corpus-specific)
// ===========================================================================
// Compare raw scores to fixed cutoffs. Free, trivially simple — but the right
// cutoff depends on the embedding model, the distance metric, and the corpus.
// A cosine of 0.30 is "unrelated" for one model and "quite similar" for
// another. Never ship these without calibrating on YOUR data.

/** 1a. Similarity floor — drop every hit below τ. */
export function similarityFloor(hits: Hit[], tau: number): Hit[] {
  return hits.filter((h) => h.score >= tau);
}

/** 1b. Top-1 floor — if even the best hit is weak, refuse outright.
 *  The cheapest possible "I don't know" gate. */
export function top1Floor(hits: Hit[], tau: number): boolean {
  return hits.length > 0 && hits[0].score >= tau;
}

/** 1c. Count-above-threshold — require at least `minCount` decent hits, so a
 *  single lucky keyword match can't carry an answer alone. */
export function countAboveThreshold(
  hits: Hit[],
  tau: number,
  minCount: number
): boolean {
  return hits.filter((h) => h.score >= tau).length >= minCount;
}

/** 1d. Calibrated threshold — derive τ from data instead of guessing.
 *  Given similarity scores of known-IRRELEVANT query/chunk pairs (a small
 *  labelled dev set), pick the given percentile (e.g. 0.95) as τ: anything
 *  scoring above it is unlikely to be background noise.
 *  This is the only defensible way to set an absolute cutoff — and it must be
 *  redone whenever the embedding model or corpus changes. */
export function calibrateThreshold(
  irrelevantPairScores: number[],
  percentile = 0.95
): number {
  const sorted = [...irrelevantPairScores].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.floor(percentile * sorted.length)
  );
  return sorted[idx];
}

/** 1e. Reranker-score floor — same rule, better input. Cross-encoder /
 *  pointwise-LLM scores (see 3.6) are far better calibrated than raw cosine,
 *  so an absolute floor on THEM transfers much better. */
export function rerankerFloor(
  hits: Hit[],
  rerankScore: (h: Hit) => number,
  tau: number
): Hit[] {
  return hits.filter((h) => rerankScore(h) >= tau);
}

// ===========================================================================
// 2. RELATIVE / DISTRIBUTIONAL SIGNALS (more robust)
// ===========================================================================
// Ignore the absolute values; look at the SHAPE of the top-K score
// distribution. Shape transfers across models and corpora much better:
// "one hit towers over the rest" means the same thing everywhere.

/** 2a. Top-1/Top-2 gap — a clear winner separates from the pack.
 *  Big gap ⇒ one chunk is distinctly relevant; near-zero gap ⇒ either many
 *  good hits (fine) or uniformly mediocre ones (combine with other signals). */
export function top1Top2Gap(hits: Hit[]): number {
  if (hits.length < 2) return hits.length === 1 ? hits[0].score : 0;
  return hits[0].score - hits[1].score;
}

/** 2b. Relative drop-off — top-1 vs the mean of the rest.
 *  Ratio ≫ 1 ⇒ the winner is a real outlier; ≈ 1 ⇒ everything scored about
 *  the same, i.e. the retriever couldn't tell the candidates apart. */
export function relativeDropoff(hits: Hit[]): number {
  if (hits.length < 2) return hits.length === 1 ? Infinity : 0;
  const rest = hits.slice(1);
  const mean = rest.reduce((s, h) => s + h.score, 0) / rest.length;
  return mean <= 0 ? Infinity : hits[0].score / mean;
}

/** 2c. Softmax entropy — how "peaked" is the score distribution?
 *  Softmax the scores (temperature sharpens/flattens), take Shannon entropy,
 *  normalise to [0,1] by dividing by log(K).
 *  ~0 ⇒ all mass on one hit (confident); ~1 ⇒ uniform (retriever is lost). */
export function softmaxEntropy(hits: Hit[], temperature = 0.1): number {
  if (hits.length < 2) return 0;
  const exps = hits.map((h) => Math.exp(h.score / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);
  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
  return entropy / Math.log(hits.length); // normalised: 0=peaked, 1=uniform
}

/** 2d. Z-score vs corpus background — is the top hit an outlier compared to
 *  how similar RANDOM chunks are to the query? Estimate μ/σ once from a
 *  random sample of the corpus (cheap, offline-able per query). z > ~3 means
 *  the top hit is genuinely special, whatever the absolute cosine says. */
export function zScoreVsBackground(
  topScore: number,
  backgroundScores: number[]
): number {
  const mu =
    backgroundScores.reduce((a, b) => a + b, 0) / backgroundScores.length;
  const variance =
    backgroundScores.reduce((s, x) => s + (x - mu) ** 2, 0) /
    backgroundScores.length;
  const sigma = Math.sqrt(variance);
  return sigma === 0 ? 0 : (topScore - mu) / sigma;
}

/** 2e. Cross-retriever agreement — Jaccard overlap of two independent top-K
 *  lists (e.g. dense + BM25). Two retrievers with different failure modes
 *  agreeing on the same chunks is strong evidence they found something real;
 *  zero overlap on an in-domain query is a red flag. */
export function crossRetrieverAgreement(
  denseIds: string[],
  keywordIds: string[]
): number {
  const a = new Set(denseIds);
  const b = new Set(keywordIds);
  const inter = [...a].filter((id) => b.has(id)).length;
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : inter / union;
}

// ===========================================================================
// 3. LLM-BASED GROUNDEDNESS CHECKS (most robust, most expensive)
// ===========================================================================
// Ask a model to READ the evidence. This catches what vector math cannot:
// a chunk can be topically similar yet not actually answer the question
// ("mentions HNSW" vs "tells you how to tune HNSW").
// Judges are injected as functions so the demo can use lexical mocks and
// production can use Claude.

export type Answerability = "YES" | "PARTIAL" | "NO";

/** 3a. Answerability judge — pre-generation gate.
 *  One LLM call: "Can these passages answer this query?" This is the single
 *  highest-value LLM check: it runs BEFORE you spend tokens generating a
 *  wrong answer. */
export function answerabilityCheck(
  query: string,
  hits: Hit[],
  judge: (query: string, passages: string[]) => Answerability
): Answerability {
  return judge(query, hits.map((h) => h.text));
}

/** 3b. Per-passage grading — pointwise filter. Grade each passage 0–10 for
 *  "does it answer the query" and drop the ones below cutoff, so junk never
 *  reaches the generator's context. (Same mechanism as 3.6's pointwise LLM
 *  rerank, used here as a gate rather than a sort.) */
export function gradePassages(
  query: string,
  hits: Hit[],
  grader: (query: string, passage: string) => number,
  cutoff = 5
): Hit[] {
  return hits.filter((h) => grader(query, h.text) >= cutoff);
}

/** 3c. Claim-level groundedness (NLI-style) — post-generation verification.
 *  Split the draft answer into claims; for each, ask "is this claim supported
 *  by the evidence?" Returns supported/unsupported claims + a groundedness
 *  ratio. Unsupported claims are hallucination candidates: strip them, hedge,
 *  or regenerate. */
export function groundednessCheck(
  answerClaims: string[],
  evidence: Hit[],
  entails: (claim: string, evidence: string[]) => boolean
): { supported: string[]; unsupported: string[]; ratio: number } {
  const texts = evidence.map((h) => h.text);
  const supported: string[] = [];
  const unsupported: string[] = [];
  for (const claim of answerClaims) {
    (entails(claim, texts) ? supported : unsupported).push(claim);
  }
  const ratio =
    answerClaims.length === 0 ? 1 : supported.length / answerClaims.length;
  return { supported, unsupported, ratio };
}

/** 3d. Citation verification — for each "fact [chunk-id]" citation in the
 *  answer, check the cited chunk actually contains/entails the fact. Catches
 *  the classic failure of citing a real chunk for a made-up statement. */
export function verifyCitations(
  citations: { claim: string; chunkId: string }[],
  hits: Hit[],
  entails: (claim: string, evidence: string[]) => boolean
): { valid: number; invalid: number } {
  let valid = 0;
  let invalid = 0;
  for (const c of citations) {
    const chunk = hits.find((h) => h.id === c.chunkId);
    if (chunk && entails(c.claim, [chunk.text])) valid++;
    else invalid++;
  }
  return { valid, invalid };
}

/** 3e. Self-consistency — sample the answer N times (temperature > 0); if the
 *  samples disagree, the model is guessing, whatever the retrieval scores
 *  said. `agreement` is the fraction of samples matching the majority answer.
 *  N× generation cost — reserve for high-stakes answers. */
export function selfConsistency(
  sampleAnswer: () => string,
  n = 3
): { majority: string; agreement: number } {
  const counts = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const a = sampleAnswer();
    counts.set(a, (counts.get(a) ?? 0) + 1);
  }
  let majority = "";
  let best = 0;
  for (const [a, c] of counts) if (c > best) [majority, best] = [a, c];
  return { majority, agreement: best / n };
}

// ===========================================================================
// 4. THE COMPLETE CONFIDENCE PIPELINE
// ===========================================================================
// Compose the signals cheapest-first with early exits. Most queries decide at
// stage 1–2 for free; only the gray zone pays for LLM calls. Result: LLM-grade
// robustness at near-vector-math average cost.

export interface PipelineConfig {
  tauHard: number; // stage 1: below this top-1 score, refuse immediately
  gapConfident: number; // stage 2: gap ≥ this ⇒ skip LLM gate
  entropyConfident: number; // stage 2: entropy ≤ this ⇒ skip LLM gate
  entropyHopeless: number; // stage 2: entropy ≥ this AND weak top-1 ⇒ refuse
  groundednessMin: number; // stage 5: below this claim-support ratio ⇒ hedge
}

export interface PipelineDeps {
  answerabilityJudge: (query: string, passages: string[]) => Answerability;
  generate: (query: string, hits: Hit[]) => { answer: string; claims: string[] };
  entails: (claim: string, evidence: string[]) => boolean;
}

export interface PipelineResult {
  verdict: Verdict;
  answer?: string;
  decidedAt: string; // which stage made the call
  trace: string[]; // human-readable log of every stage's reading
}

export function confidencePipeline(
  query: string,
  hits: Hit[],
  cfg: PipelineConfig,
  deps: PipelineDeps
): PipelineResult {
  const trace: string[] = [];

  // --- Stage 1: absolute floor (free, instant) ----------------------------
  const top1 = hits[0]?.score ?? 0;
  trace.push(`stage 1 [absolute]      top-1=${top1.toFixed(3)} vs τ_hard=${cfg.tauHard}`);
  if (!top1Floor(hits, cfg.tauHard)) {
    return { verdict: "REFUSE", decidedAt: "stage 1: absolute floor", trace };
  }

  // --- Stage 2: distributional shape (free) --------------------------------
  const gap = top1Top2Gap(hits);
  const entropy = softmaxEntropy(hits);
  trace.push(
    `stage 2 [distributional] gap=${gap.toFixed(3)} entropy=${entropy.toFixed(3)}`
  );
  let needLlmGate = true;
  if (gap >= cfg.gapConfident || entropy <= cfg.entropyConfident) {
    trace.push("stage 2: clear winner — skipping LLM gate");
    needLlmGate = false; // confident: go straight to generation
  } else if (entropy >= cfg.entropyHopeless && top1 < cfg.tauHard * 1.5) {
    return {
      verdict: "REFUSE",
      decidedAt: "stage 2: flat distribution + weak scores",
      trace,
    };
  }

  // --- Stage 3: LLM answerability gate (only for the gray zone) -----------
  let hedging = false;
  if (needLlmGate) {
    const verdict = answerabilityCheck(query, hits, deps.answerabilityJudge);
    trace.push(`stage 3 [LLM gate]      answerability=${verdict}`);
    if (verdict === "NO") {
      return { verdict: "REFUSE", decidedAt: "stage 3: LLM answerability", trace };
    }
    hedging = verdict === "PARTIAL";
  }

  // --- Stage 4: generate with citations ------------------------------------
  const { answer, claims } = deps.generate(query, hits);
  trace.push(`stage 4 [generate]      ${claims.length} claim(s)`);

  // --- Stage 5: groundedness verification ----------------------------------
  const g = groundednessCheck(claims, hits, deps.entails);
  trace.push(
    `stage 5 [groundedness]  ${g.supported.length}/${claims.length} claims supported (ratio=${g.ratio.toFixed(2)})`
  );
  if (g.ratio < cfg.groundednessMin) hedging = true;

  return {
    verdict: hedging ? "HEDGE" : "ANSWER",
    answer,
    decidedAt: hedging
      ? "stage 5: partial support → hedge"
      : "stage 5: fully grounded",
    trace,
  };
}
