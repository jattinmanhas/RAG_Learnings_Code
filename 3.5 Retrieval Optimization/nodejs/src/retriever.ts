/**
 * ============================================================================
 *  RETRIEVAL OPTIMIZATION FOR RAG  (Node.js / TypeScript)
 * ============================================================================
 *
 *  WHAT THIS FILE COVERS
 *  ---------------------
 *  Sections 3.4.1 and 3.4.2 answered "what do I embed and how big is the
 *  context I return?" (parent-child, small-to-big). This file answers the next
 *  question every production RAG system hits: "given a corpus of chunks, how do
 *  I pull back the RIGHT ones — quickly, filtered, and tunably?"
 *
 *  It demonstrates the four levers you reach for, in the order you reach them:
 *
 *    1. TOP-K RETRIEVAL     — the baseline. Embed the query, rank every chunk
 *                             by cosine similarity, keep the best K. Everything
 *                             else refines this.
 *
 *    2. METADATA FILTERING  — narrow the candidate set BEFORE the vector search
 *                             using structured fields: tenant, source, language,
 *                             date, access level. This is about correctness and
 *                             security, not just relevance — never return a
 *                             chunk the user isn't allowed to see.
 *
 *    3. HYBRID RETRIEVAL    — dense vectors capture meaning ("car" ~
 *                             "automobile") but are blind to exact tokens (SKUs,
 *                             error codes, rare names). Sparse keyword search
 *                             (BM25) is the opposite. Hybrid runs BOTH and fuses
 *                             the rankings for semantic recall AND exact-match
 *                             precision.
 *
 *    4. RETRIEVAL TUNING    — none of the above has one right setting. topK, the
 *                             dense/sparse mix (alpha), the RRF constant, and a
 *                             score floor are all knobs. Every knob here is an
 *                             explicit parameter, and the demo (index.ts) shows
 *                             how turning each one changes the results.
 *
 *  DESIGN NOTES
 *  ------------
 *   • Dense scores come from an injectable EmbedFn (mock in the demo, real
 *     provider in production — see the comment on EmbedFn).
 *   • Sparse scores come from a real, self-contained BM25 implementation so the
 *     hybrid fusion is honest, not hand-waved.
 *   • Fusion supports the two approaches you'll see in the wild:
 *       - Reciprocal Rank Fusion (RRF): rank-based, scale-free. Safe default.
 *       - Weighted score fusion (alpha): blends normalised dense & sparse
 *         scores. More control, but you own the normalisation.
 *   • Everything is in-memory for clarity. Inline SQL comments show the
 *     pgvector equivalent you'd run in production.
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// CORE TYPES
// ---------------------------------------------------------------------------

/** One retrievable unit — a passage plus the metadata you filter on. */
export interface Chunk {
  /** Stable identifier, e.g. "doc7-chunk-3". */
  id: string;
  /** The passage that gets embedded and returned. */
  text: string;
  /** Structured fields you can filter on. */
  metadata: Record<string, string>;
}

/**
 * A chunk paired with a score plus the individual dense/sparse components that
 * produced it. Surfacing the components is what makes TUNING possible: you can
 * see whether a result won on meaning, on keywords, or on both.
 */
export interface Scored {
  chunk: Chunk;
  /** Final score used for ranking (fused, in hybrid mode). */
  score: number;
  /** Cosine similarity component (0 if not computed). */
  denseScore: number;
  /** BM25 component (0 if not computed). */
  sparseScore: number;
  /** 1-based rank in the dense list (0 = not present). */
  denseRank: number;
  /** 1-based rank in the sparse list (0 = not present). */
  sparseRank: number;
}

/**
 * Converts text into a vector. Inject your real provider here.
 *
 * @example — Anthropic pairs with Voyage AI for embeddings (Claude itself does
 * not expose an embeddings endpoint):
 *   import Anthropic from "@anthropic-ai/sdk";   // for generation
 *   import VoyageAI from "voyageai";             // for embeddings
 *   const voyage = new VoyageAI();
 *   const embed: EmbedFn = async (text) => {
 *     const res = await voyage.embed({ input: [text], model: "voyage-3" });
 *     return res.data[0].embedding;
 *   };
 */
export type EmbedFn = (text: string) => Promise<number[]>;

/**
 * A metadata predicate: return true to KEEP the chunk as a candidate.
 * Compose these to express "language == en AND access <= user's level".
 */
export type Filter = (chunk: Chunk) => boolean;

// ---------------------------------------------------------------------------
// METADATA FILTERING (topic 2)
// ---------------------------------------------------------------------------
//
// Filtering is applied to candidates BEFORE scoring so the vector search never
// wastes work on — or returns — chunks the user shouldn't see. In pgvector this
// is a WHERE clause on the same SELECT as the vector search:
//
//   SELECT id, text, 1 - (embedding <=> $q) AS score
//   FROM   chunks
//   WHERE  metadata->>'language' = 'en'
//     AND  (metadata->>'access_level')::int <= $userLevel
//   ORDER  BY embedding <=> $q
//   LIMIT  $k;
//
// One query lets Postgres use a partial/btree index on the metadata column, so
// you don't scan vectors you're about to discard.

/** Keeps every chunk — the "no filter" default. */
export const matchAll: Filter = () => true;

/** Keeps chunks whose metadata[key] exactly equals value. */
export function fieldEquals(key: string, value: string): Filter {
  return (c) => c.metadata[key] === value;
}

/** Keeps chunks whose metadata[key] is one of the allowed values. */
export function fieldIn(key: string, ...values: string[]): Filter {
  const allowed = new Set(values);
  return (c) => allowed.has(c.metadata[key]);
}

/**
 * Composes filters with logical AND — a chunk must satisfy every filter.
 * This is how you stack a relevance filter on top of a security filter.
 */
export function and(...filters: Filter[]): Filter {
  return (c) => filters.every((f) => f(c));
}

/** Returns the subset of chunks that pass the predicate. */
function applyFilter(chunks: Chunk[], filter: Filter): Chunk[] {
  return chunks.filter(filter);
}

// ---------------------------------------------------------------------------
// DENSE RETRIEVAL (topic 1: Top-K)
// ---------------------------------------------------------------------------

/**
 * Embed the query and every candidate, then rank candidates by cosine
 * similarity. Returned best-first; the caller truncates to K.
 *
 * In production the loop is a single pgvector query — the embeddings already
 * live in the table, so you don't re-embed the corpus per request. Here we
 * embed on the fly because the mock embedder is free.
 */
async function denseSearch(
  query: string,
  candidates: Chunk[],
  embed: EmbedFn
): Promise<Scored[]> {
  const queryVec = await embed(query);

  const scored: Scored[] = await Promise.all(
    candidates.map(async (chunk) => {
      const vec = await embed(chunk.text);
      const sim = cosineSimilarity(queryVec, vec);
      return {
        chunk,
        score: sim,
        denseScore: sim,
        sparseScore: 0,
        denseRank: 0,
        sparseRank: 0,
      };
    })
  );

  scored.sort((a, b) => b.denseScore - a.denseScore);
  scored.forEach((s, i) => (s.denseRank = i + 1));
  return scored;
}

/** Cosine similarity of two equal-length vectors; range [−1, 1]. */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// ---------------------------------------------------------------------------
// SPARSE RETRIEVAL — BM25 (the keyword half of hybrid)
// ---------------------------------------------------------------------------
//
// BM25 is the workhorse lexical ranking function behind Elasticsearch, Lucene,
// and Postgres full-text search. It scores a chunk for a query by summing, over
// each query term, that term's IDF (rare terms count more) times a saturating
// term-frequency factor (the 10th occurrence adds little over the 2nd) with a
// length normalisation (long chunks don't win just by being long).
//
// We implement it directly so the hybrid demo is genuine. In production you'd
// let Postgres/Elasticsearch compute this; the fusion logic below is identical.

/** The two classic BM25 knobs — themselves part of retrieval TUNING. */
export interface BM25Params {
  /** Term-frequency saturation (typical 1.2–2.0). */
  k1: number;
  /** Length-normalisation strength (typical 0.75; 0 disables). */
  b: number;
}

export function defaultBM25(): BM25Params {
  return { k1: 1.5, b: 0.75 };
}

/**
 * Lowercase and split on non-alphanumerics — the minimal viable analyzer.
 * Production systems add stemming, stop-word removal, and synonyms; the scoring
 * math is unchanged.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 0);
}

/**
 * Precomputes everything term-frequency-based so scoring a query is cheap.
 * Built once at ingest, reused for every query.
 */
class BM25Index {
  private docFreq = new Map<string, number>();
  private tf: Map<string, number>[] = [];
  private lens: number[] = [];
  private avgLen = 0;
  private n = 0;

  constructor(chunks: Chunk[], private params: BM25Params) {
    let totalLen = 0;
    this.n = chunks.length;
    for (const c of chunks) {
      const counts = new Map<string, number>();
      const toks = tokenize(c.text);
      for (const t of toks) counts.set(t, (counts.get(t) ?? 0) + 1);
      this.tf.push(counts);
      this.lens.push(toks.length);
      totalLen += toks.length;
      for (const term of counts.keys()) {
        this.docFreq.set(term, (this.docFreq.get(term) ?? 0) + 1);
      }
    }
    if (this.n > 0) this.avgLen = totalLen / this.n;
  }

  /** BM25 "probabilistic" IDF with +1 to keep it non-negative. */
  private idf(term: string): number {
    const df = this.docFreq.get(term) ?? 0;
    if (df === 0) return 0;
    return Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
  }

  /** BM25 score of chunk i for the given query terms. */
  scoreDoc(i: number, queryTerms: string[]): number {
    if (i < 0 || i >= this.n) return 0;
    const { k1, b } = this.params;
    let score = 0;
    for (const term of queryTerms) {
      const f = this.tf[i].get(term) ?? 0;
      if (f === 0) continue;
      const norm = 1 - b + (b * this.lens[i]) / this.avgLen;
      score += (this.idf(term) * (f * (k1 + 1))) / (f + k1 * norm);
    }
    return score;
  }
}

/**
 * Rank candidates by BM25 against the query. Takes the (already
 * metadata-filtered) candidate subset plus a position lookup so it can reuse
 * the precomputed term frequencies.
 */
function sparseSearch(
  query: string,
  candidates: Chunk[],
  index: BM25Index,
  pos: Map<string, number>
): Scored[] {
  const terms = tokenize(query);
  const scored: Scored[] = candidates
    .filter((c) => pos.has(c.id))
    .map((chunk) => {
      const s = index.scoreDoc(pos.get(chunk.id)!, terms);
      return {
        chunk,
        score: s,
        denseScore: 0,
        sparseScore: s,
        denseRank: 0,
        sparseRank: 0,
      };
    });
  scored.sort((a, b) => b.sparseScore - a.sparseScore);
  scored.forEach((s, i) => (s.sparseRank = i + 1));
  return scored;
}

// ---------------------------------------------------------------------------
// FUSION — combining dense + sparse (topic 3: Hybrid)
// ---------------------------------------------------------------------------

/** How the two ranked lists are combined. */
export enum FusionMode {
  /**
   * Reciprocal Rank Fusion. Combines by RANK, not score, so it needs no score
   * normalisation and is robust when dense and sparse scores live on wildly
   * different scales. The production-safe default.
   */
  RRF = "rrf",
  /**
   * Blend min-max-normalised dense & sparse SCORES with a weight alpha
   * (alpha=1 → pure dense, alpha=0 → pure sparse). More control, but you own
   * the normalisation and it's sensitive to score outliers.
   */
  Weighted = "weighted",
}

/**
 * Reciprocal Rank Fusion:
 *
 *   score(chunk) = Σ  1 / (rrfK + rank_in_list)
 *
 * summed over the dense and sparse lists the chunk appears in. rrfK (typically
 * 60) damps the contribution of top ranks so no single list dominates. A chunk
 * ranking well in BOTH lists beats one great in only one — the hybrid win.
 */
function fuseRRF(dense: Scored[], sparse: Scored[], rrfK: number): Scored[] {
  const k = rrfK > 0 ? rrfK : 60;
  const agg = new Map<string, Scored>();

  const accumulate = (list: Scored[], isDense: boolean) => {
    list.forEach((s, rank) => {
      let cur = agg.get(s.chunk.id);
      if (!cur) {
        cur = blank(s.chunk);
        agg.set(s.chunk.id, cur);
      }
      cur.score += 1 / (k + (rank + 1));
      if (isDense) {
        cur.denseScore = s.denseScore;
        cur.denseRank = rank + 1;
      } else {
        cur.sparseScore = s.sparseScore;
        cur.sparseRank = rank + 1;
      }
    });
  };
  accumulate(dense, true);
  accumulate(sparse, false);

  return sortAgg(agg);
}

/**
 * Min-max normalises each list to [0,1] and blends:
 *
 *   score = alpha * denseNorm + (1 - alpha) * sparseNorm
 */
function fuseWeighted(dense: Scored[], sparse: Scored[], alpha: number): Scored[] {
  const a = clamp01(alpha);
  const dNorm = minMaxNormalize(dense, (s) => s.denseScore);
  const sNorm = minMaxNormalize(sparse, (s) => s.sparseScore);

  const agg = new Map<string, Scored>();
  const ensure = (s: Scored): Scored => {
    let cur = agg.get(s.chunk.id);
    if (!cur) {
      cur = blank(s.chunk);
      agg.set(s.chunk.id, cur);
    }
    return cur;
  };
  dense.forEach((s, i) => {
    const cur = ensure(s);
    cur.denseScore = s.denseScore;
    cur.denseRank = i + 1;
    cur.score += a * (dNorm.get(s.chunk.id) ?? 0);
  });
  sparse.forEach((s, i) => {
    const cur = ensure(s);
    cur.sparseScore = s.sparseScore;
    cur.sparseRank = i + 1;
    cur.score += (1 - a) * (sNorm.get(s.chunk.id) ?? 0);
  });
  return sortAgg(agg);
}

function minMaxNormalize(list: Scored[], get: (s: Scored) => number): Map<string, number> {
  const out = new Map<string, number>();
  if (list.length === 0) return out;
  let min = Infinity,
    max = -Infinity;
  for (const s of list) {
    const v = get(s);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min;
  for (const s of list) {
    out.set(s.chunk.id, span === 0 ? 0 : (get(s) - min) / span);
  }
  return out;
}

function blank(chunk: Chunk): Scored {
  return { chunk, score: 0, denseScore: 0, sparseScore: 0, denseRank: 0, sparseRank: 0 };
}

function sortAgg(agg: Map<string, Scored>): Scored[] {
  return [...agg.values()].sort((a, b) =>
    b.score !== a.score ? b.score - a.score : a.chunk.id.localeCompare(b.chunk.id)
  );
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// ---------------------------------------------------------------------------
// THE STORE + PUBLIC RETRIEVAL API (topic 4: everything is a tunable knob)
// ---------------------------------------------------------------------------

/**
 * Holds the corpus, the precomputed BM25 index, and a chunk-id → position
 * lookup. Build once, query many times.
 */
export class Store {
  private bm25: BM25Index;
  private pos = new Map<string, number>();

  constructor(private chunks: Chunk[], bm25Cfg: BM25Params = defaultBM25()) {
    chunks.forEach((c, i) => this.pos.set(c.id, i));
    this.bm25 = new BM25Index(chunks, bm25Cfg);
  }

  /**
   * Full optimised pipeline:
   *
   *   metadata filter → dense (+ sparse) search → fuse → threshold → top-K
   *
   * Every stage is governed by RetrieveOptions, so the same method covers plain
   * top-K, filtered search, and full hybrid retrieval depending on config.
   */
  async retrieve(query: string, embed: EmbedFn, opts: RetrieveOptions = {}): Promise<Scored[]> {
    const o = { ...defaultOptions(), ...opts };

    // Stage 1 — metadata filter (topic 2): shrink the candidate set first.
    const candidates = applyFilter(this.chunks, o.filter);
    if (candidates.length === 0) return [];

    // Stage 2 — dense retrieval (topic 1), truncated to candidateK.
    let dense = await denseSearch(query, candidates, embed);
    dense = dense.slice(0, o.candidateK);

    // Dense-only path: no fusion, just threshold + top-K.
    if (!o.hybrid) return finalize(dense, o);

    // Stage 3 — sparse retrieval (topic 3), same candidates + candidateK.
    let sparse = sparseSearch(query, candidates, this.bm25, this.pos);
    sparse = sparse.slice(0, o.candidateK);

    // Stage 4 — fuse the two rankings (topic 3).
    const fused =
      o.fusion === FusionMode.Weighted
        ? fuseWeighted(dense, sparse, o.alpha)
        : fuseRRF(dense, sparse, o.rrfK);

    // Stage 5 — threshold + top-K (topic 4).
    return finalize(fused, o);
  }
}

/**
 * Every retrieval knob in one place. This IS "retrieval tuning": each field is
 * a lever, and index.ts sweeps them to show the effect.
 */
export interface RetrieveOptions {
  /** How many results to return (topic 1). */
  topK?: number;
  /** Metadata predicate; defaults to matchAll (topic 2). */
  filter?: Filter;
  /** false = dense only; true = dense + BM25 (topic 3). */
  hybrid?: boolean;
  /** RRF or weighted, when hybrid (topic 3). */
  fusion?: FusionMode;
  /** Dense weight for weighted fusion (topic 4). */
  alpha?: number;
  /** RRF damping constant, default 60 (topic 4). */
  rrfK?: number;
  /** Drop results whose final score < threshold (topic 4). */
  threshold?: number;
  /** Per-retriever candidate cap feeding fusion — the retrieve-wide-then-narrow knob. */
  candidateK?: number;
}

/** Sensible production starting point: hybrid + RRF, top 5. */
export function defaultOptions(): Required<RetrieveOptions> {
  return {
    topK: 5,
    filter: matchAll,
    hybrid: true,
    fusion: FusionMode.RRF,
    alpha: 0.5,
    rrfK: 60,
    threshold: 0,
    candidateK: 20,
  };
}

/** Apply the score threshold, then truncate to topK. */
function finalize(list: Scored[], o: Required<RetrieveOptions>): Scored[] {
  return list.filter((s) => s.score >= o.threshold).slice(0, o.topK);
}
