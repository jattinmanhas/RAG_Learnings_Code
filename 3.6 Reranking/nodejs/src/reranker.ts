/**
 * reranker.ts — every common reranking strategy, dependency-free.
 *
 * Reranking is STAGE 2 of retrieval. Stage 1 (bi-encoder / BM25 / hybrid)
 * casts a wide, cheap net over the whole corpus and returns top-K candidates.
 * Stage 2 re-scores ONLY those K candidates with a slower, more accurate
 * model and reorders them before context construction.
 *
 *   corpus (1M docs) --stage 1: bi-encoder--> top 50 --stage 2: reranker--> top 5
 *
 * Why the split works:
 *   - A bi-encoder embeds query and document SEPARATELY, then compares
 *     vectors. Fast (documents pre-embedded offline), but the model never
 *     sees query and document together, so it misses fine-grained relevance.
 *   - A cross-encoder (or LLM judge) reads query + document TOGETHER in one
 *     forward pass. Far more accurate, but O(K) model calls per query — you
 *     can only afford it on a shortlist.
 *
 * Strategies implemented here:
 *   1. Cross-encoder reranking      — the production standard.
 *   2. Late interaction (ColBERT)   — token-level MaxSim; the middle ground.
 *   3. LLM pointwise reranking      — score each doc 0–10 independently.
 *   4. LLM pairwise reranking       — "which of these two is better?" duels.
 *   5. LLM listwise reranking       — RankGPT-style: rank the whole list at once.
 *   6. MMR                          — diversity, not just relevance.
 *   7. RRF                          — fuse ranked lists from several retrievers.
 *   8. Business-rule reranking      — recency decay + source-authority boosts.
 *
 * All model calls are injected as callbacks (CrossEncoderFn, LlmJudgeFn, …)
 * so this file stays dependency-free. index.ts provides lexical mocks;
 * production swaps in Cohere Rerank, Voyage rerank-2, a BGE cross-encoder,
 * or a Claude/GPT judge without touching this logic.
 */

// ---------------------------------------------------------------------------
// TYPES
// ---------------------------------------------------------------------------

export interface Chunk {
  id: string;
  text: string;
  metadata: Record<string, string>;
}

/** A chunk with a score attached — the currency of every reranker. */
export interface Scored {
  chunk: Chunk;
  score: number;
}

/** Bi-encoder style embedding: one vector per text. */
export type EmbedFn = (text: string) => number[];

/**
 * Cross-encoder: sees query and document together, returns a relevance
 * score. In production this is one forward pass of a model like
 * BAAI/bge-reranker-v2-m3, or one API call to Cohere Rerank / Voyage rerank.
 */
export type CrossEncoderFn = (query: string, doc: string) => number;

/**
 * LLM judge for pointwise scoring: returns relevance 0–10.
 * In production: a small/cheap LLM with a rubric prompt, temperature 0.
 */
export type LlmJudgeFn = (query: string, doc: string) => number;

/**
 * LLM judge for pairwise comparison: returns the winner ("a" or "b").
 * In production: "Query: … Passage A: … Passage B: … Which passage answers
 * the query better? Reply A or B."
 */
export type LlmDuelFn = (query: string, a: string, b: string) => "a" | "b";

/**
 * LLM listwise ranker: given the query and all candidate texts, returns the
 * indices in best-first order. This is the RankGPT pattern — one prompt
 * containing numbered passages, model replies "[3] > [1] > [5] > …".
 */
export type LlmListRankFn = (query: string, docs: string[]) => number[];

/** Token-level embedder for late interaction: one vector PER TOKEN. */
export type TokenEmbedFn = (text: string) => number[][];

// ---------------------------------------------------------------------------
// SHARED MATH
// ---------------------------------------------------------------------------

export function cosine(a: number[], b: number[]): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const byScoreDesc = (a: Scored, b: Scored) => b.score - a.score;

// ---------------------------------------------------------------------------
// STAGE 1 — BI-ENCODER RETRIEVAL (the thing we rerank)
// ---------------------------------------------------------------------------
// Included so the demo is self-contained: this is the fast, coarse first
// pass. Note it compares one query vector against pre-computed doc vectors —
// no model ever reads query and document side by side.

export function biEncoderRetrieve(
  query: string,
  corpus: Chunk[],
  embed: EmbedFn,
  topK: number
): Scored[] {
  const qv = embed(query);
  return corpus
    .map((chunk) => ({ chunk, score: cosine(qv, embed(chunk.text)) }))
    .sort(byScoreDesc)
    .slice(0, topK);
}

// ---------------------------------------------------------------------------
// 1. CROSS-ENCODER RERANKING — the production standard
// ---------------------------------------------------------------------------
// One scoring call per candidate: score = crossEncode(query, doc). The pair
// is processed jointly, so the model can see that "it" in the document
// refers to the thing the query asks about, that a negation flips relevance,
// that the doc mentions the query terms but answers a different question.
//
// Cost: K model calls (or one batched call). Never run it over the corpus.

export function crossEncoderRerank(
  query: string,
  candidates: Scored[],
  crossEncode: CrossEncoderFn,
  topN: number
): Scored[] {
  return candidates
    .map(({ chunk }) => ({ chunk, score: crossEncode(query, chunk.text) }))
    .sort(byScoreDesc)
    .slice(0, topN);
}

// ---------------------------------------------------------------------------
// 2. LATE INTERACTION (ColBERT-style MaxSim)
// ---------------------------------------------------------------------------
// The middle ground between bi- and cross-encoders. Documents are encoded
// offline like a bi-encoder — but into one vector PER TOKEN instead of one
// per document. At query time, each query token finds its best-matching
// document token (MaxSim), and those maxima are summed:
//
//   score(q, d) = Σ_{i ∈ q tokens} max_{j ∈ d tokens} sim(q_i, d_j)
//
// Fine-grained token-level matching (close to cross-encoder quality) with
// precomputable document representations (close to bi-encoder speed). The
// price is storage: ~100 vectors per document instead of 1.

export function lateInteractionRerank(
  query: string,
  candidates: Scored[],
  tokenEmbed: TokenEmbedFn,
  topN: number
): Scored[] {
  const qTokens = tokenEmbed(query);
  return candidates
    .map(({ chunk }) => {
      const dTokens = tokenEmbed(chunk.text);
      let score = 0;
      for (const qv of qTokens) {
        let best = 0;
        for (const dv of dTokens) best = Math.max(best, cosine(qv, dv));
        score += best; // MaxSim: each query token keeps only its best doc-token match
      }
      return { chunk, score: qTokens.length ? score / qTokens.length : 0 };
    })
    .sort(byScoreDesc)
    .slice(0, topN);
}

// ---------------------------------------------------------------------------
// 3. LLM POINTWISE RERANKING
// ---------------------------------------------------------------------------
// Ask an LLM to grade each (query, doc) pair independently on a rubric
// (0–10). Simple, parallelizable, and scores are absolute — usable as a
// confidence threshold ("if the best doc scores < 4, say I don't know").
// Weakness: LLMs are inconsistent graders; a 7 for one doc and a 6 for
// another may not reflect a real preference.

export function llmPointwiseRerank(
  query: string,
  candidates: Scored[],
  judge: LlmJudgeFn,
  topN: number
): Scored[] {
  return candidates
    .map(({ chunk }) => ({ chunk, score: judge(query, chunk.text) }))
    .sort(byScoreDesc)
    .slice(0, topN);
}

// ---------------------------------------------------------------------------
// 4. LLM PAIRWISE RERANKING
// ---------------------------------------------------------------------------
// Comparative judgments ("is A better than B?") are more reliable than
// absolute grades, at the cost of more calls. Full round-robin is O(K²)
// duels; each win earns a point and candidates are ranked by wins.
// Production systems cut the cost with a single bubble/insertion pass
// (O(K)) or a tournament bracket (O(K log K)).

export function llmPairwiseRerank(
  query: string,
  candidates: Scored[],
  duel: LlmDuelFn,
  topN: number
): Scored[] {
  const wins = new Map<string, number>(candidates.map((c) => [c.chunk.id, 0]));
  for (let i = 0; i < candidates.length; i++) {
    for (let j = i + 1; j < candidates.length; j++) {
      const a = candidates[i].chunk;
      const b = candidates[j].chunk;
      const winner = duel(query, a.text, b.text) === "a" ? a : b;
      wins.set(winner.id, (wins.get(winner.id) ?? 0) + 1);
    }
  }
  return candidates
    .map(({ chunk }) => ({ chunk, score: wins.get(chunk.id) ?? 0 }))
    .sort(byScoreDesc)
    .slice(0, topN);
}

// ---------------------------------------------------------------------------
// 5. LLM LISTWISE RERANKING (RankGPT pattern)
// ---------------------------------------------------------------------------
// Put ALL candidates in one prompt as numbered passages and ask the model to
// output a ranking ("[2] > [5] > [1] > …"). One call, and the model sees the
// candidates in context of each other — often the best quality per dollar.
// Weaknesses: position bias (models favor early passages — mitigate by
// shuffling), and the whole list must fit the context window (for large K,
// rank overlapping windows and merge — the sliding-window RankGPT trick).

export function llmListwiseRerank(
  query: string,
  candidates: Scored[],
  rankList: LlmListRankFn,
  topN: number
): Scored[] {
  const order = rankList(
    query,
    candidates.map((c) => c.chunk.text)
  );
  const ranked: Scored[] = [];
  for (const idx of order) {
    if (idx >= 0 && idx < candidates.length) {
      // Score by rank position so downstream code still sees "higher = better".
      ranked.push({ chunk: candidates[idx].chunk, score: candidates.length - ranked.length });
    }
  }
  return ranked.slice(0, topN);
}

// ---------------------------------------------------------------------------
// 6. MMR — MAXIMAL MARGINAL RELEVANCE (diversity reranking)
// ---------------------------------------------------------------------------
// Every reranker above optimizes pure relevance, so the top-N is often five
// near-duplicates of the same paragraph. MMR greedily picks the next doc
// that is relevant to the query AND different from what's already picked:
//
//   MMR = λ · sim(doc, query) − (1 − λ) · max sim(doc, already-selected)
//
// λ = 1 → pure relevance; λ = 0 → pure diversity. 0.5–0.7 is a sane default.
// Typically run AFTER a relevance reranker, on its output.

export function mmrRerank(
  query: string,
  candidates: Scored[],
  embed: EmbedFn,
  lambda: number,
  topN: number
): Scored[] {
  const qv = embed(query);
  const pool = candidates.map((c) => ({
    ...c,
    vec: embed(c.chunk.text),
    rel: cosine(qv, embed(c.chunk.text)),
  }));
  const selected: typeof pool = [];

  while (selected.length < topN && pool.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      const maxSimToSelected = selected.length
        ? Math.max(...selected.map((s) => cosine(pool[i].vec, s.vec)))
        : 0;
      const mmr = lambda * pool[i].rel - (1 - lambda) * maxSimToSelected;
      if (mmr > bestScore) {
        bestScore = mmr;
        bestIdx = i;
      }
    }
    const [picked] = pool.splice(bestIdx, 1);
    selected.push({ ...picked, score: bestScore });
  }
  return selected.map(({ chunk, score }) => ({ chunk, score }));
}

// ---------------------------------------------------------------------------
// 7. RRF — RECIPROCAL RANK FUSION (reranking across retrievers)
// ---------------------------------------------------------------------------
// When several retrievers each return a ranked list (dense, BM25, a second
// embedding model…), RRF merges them using only RANK positions — no score
// normalization needed, which is exactly why it's robust:
//
//   RRF(doc) = Σ_lists 1 / (k + rank_in_list)      k ≈ 60 by convention
//
// Also covered in 3.5 as hybrid-retrieval fusion; it appears here because
// fusing ranked lists IS a form of reranking, and it's the zero-model-call
// baseline every learned reranker must beat.

export function rrfRerank(lists: Scored[][], k: number, topN: number): Scored[] {
  const fused = new Map<string, Scored>();
  for (const list of lists) {
    list.forEach(({ chunk }, rank) => {
      const prev = fused.get(chunk.id);
      const add = 1 / (k + rank + 1);
      fused.set(chunk.id, { chunk, score: (prev?.score ?? 0) + add });
    });
  }
  return [...fused.values()].sort(byScoreDesc).slice(0, topN);
}

// ---------------------------------------------------------------------------
// 8. BUSINESS-RULE RERANKING (recency, authority, boosts)
// ---------------------------------------------------------------------------
// Pure relevance is not the whole product. A support bot should prefer the
// current docs page over a 2019 forum thread even if both match. This layer
// multiplies a relevance score by deterministic, explainable factors:
//
//   final = relevance × recencyDecay(ageDays) × sourceWeight(source)
//
// Runs LAST, on already-reranked results. Keep it multiplicative and gentle
// (weights near 1.0) so business rules tilt ties rather than overrule
// relevance entirely.

export interface BusinessRules {
  /** e.g. { docs: 1.2, changelog: 1.0, forum: 0.7 } — unlisted sources get 1.0 */
  sourceWeights: Record<string, number>;
  /** score halves every `halfLifeDays` of document age (metadata.ageDays) */
  halfLifeDays: number;
}

export function businessRuleRerank(
  candidates: Scored[],
  rules: BusinessRules,
  topN: number
): Scored[] {
  return candidates
    .map(({ chunk, score }) => {
      const ageDays = Number(chunk.metadata.ageDays ?? "0");
      const recency = Math.pow(0.5, ageDays / rules.halfLifeDays);
      const authority = rules.sourceWeights[chunk.metadata.source] ?? 1.0;
      return { chunk, score: score * recency * authority };
    })
    .sort(byScoreDesc)
    .slice(0, topN);
}
