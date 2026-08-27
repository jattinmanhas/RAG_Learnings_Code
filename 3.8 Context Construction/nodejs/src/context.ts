// context.ts — Context construction for RAG.
//
// Retrieval hands you a ranked list of chunks. The model sees a *string*.
// Everything between those two facts is context construction:
//
//   1. Context Packing      — fit maximum relevant content in the token budget
//   2. Context Ordering     — best evidence at the top AND bottom, not the middle
//   3. Context Compression  — summarise / trim chunks before injection
//   4. Citation formatting  — label evidence so the answer can point back at it
//   + the Complete Context Construction Pipeline that composes all four.
//
// Everything is model-agnostic: token counting and LLM summarisation come in
// as injected functions, so the mocks in index.ts can be swapped for a real
// tokenizer and a real Claude call without touching this file.

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

/** A retrieved chunk plus the provenance a citation will need. */
export interface Chunk {
  id: string;
  text: string;
  score: number; // relevance from retrieval/reranking (higher = better)
  source: string; // file path, URL or document title — shown in the citation
  loc?: string; // location inside the source ("L20-L48", "§3.2", "p.7")
  ordinal?: number; // position of this chunk inside its source document
}

/** Rough token counter. Swap for tiktoken / the Anthropic count-tokens API. */
export type TokenCounter = (text: string) => number;

// ===========================================================================
// 1. CONTEXT PACKING — fitting maximum relevant content within token limits
// ===========================================================================
// The context window is a budget, not a container: the prompt scaffolding, the
// query, the conversation history and the *reserved output* all spend from the
// same pool. Packing is the knapsack problem — maximise retrieved value under
// a hard token ceiling.

/** 1a. Work out how many tokens are actually left for evidence.
 *  The single most common production bug in RAG is packing to the model's
 *  context size and then getting truncated because the answer had nowhere to
 *  go. Always subtract the reservations first. */
export function evidenceBudget(opts: {
  contextWindow: number; // model's total window
  systemTokens: number; // system prompt + citation instructions
  queryTokens: number; // the user question (+ chat history)
  answerReserve: number; // room the model needs to WRITE the answer
  safetyMargin?: number; // tokenizer disagreement, formatting overhead
}): number {
  const margin = opts.safetyMargin ?? 128;
  return Math.max(
    0,
    opts.contextWindow -
      opts.systemTokens -
      opts.queryTokens -
      opts.answerReserve -
      margin
  );
}

/** 1b. Greedy packing — take chunks in relevance order while they fit.
 *  Note the `continue` (not `break`): a single oversized chunk must not stop
 *  three smaller, still-relevant chunks from getting in. */
export function packGreedy(
  chunks: Chunk[],
  budget: number,
  count: TokenCounter
): Chunk[] {
  const out: Chunk[] = [];
  let used = 0;
  for (const c of [...chunks].sort((a, b) => b.score - a.score)) {
    const t = count(c.text);
    if (used + t > budget) continue; // skip, don't stop
    out.push(c);
    used += t;
  }
  return out;
}

/** 1c. Density packing — the knapsack view. Rank by value *per token*
 *  (score / tokens) instead of raw score, so a 40-token chunk scoring 0.7
 *  beats a 900-token chunk scoring 0.8. This fits more distinct evidence in
 *  the same budget; the trade-off is that it favours short chunks, so keep the
 *  top-1 hit pinned by relevance (`pinTop`) or you can lose the best answer. */
export function packByDensity(
  chunks: Chunk[],
  budget: number,
  count: TokenCounter,
  pinTop = 1
): Chunk[] {
  const byScore = [...chunks].sort((a, b) => b.score - a.score);
  const pinned = byScore.slice(0, pinTop);
  const rest = byScore.slice(pinTop);

  const out: Chunk[] = [];
  let used = 0;
  for (const c of pinned) {
    const t = count(c.text);
    if (used + t <= budget) {
      out.push(c);
      used += t;
    }
  }
  const dense = rest
    .map((c) => ({ c, density: c.score / Math.max(1, count(c.text)) }))
    .sort((a, b) => b.density - a.density);
  for (const { c } of dense) {
    const t = count(c.text);
    if (used + t > budget) continue;
    out.push(c);
    used += t;
  }
  return out;
}

/** 1d. Near-duplicate removal — the cheapest way to buy budget.
 *  Overlapping windows (3.2) and multi-query retrieval (3.3) routinely return
 *  the same paragraph two or three times. Duplicated evidence also biases the
 *  model: it reads repetition as importance. Jaccard over word sets is a crude
 *  but effective stand-in for a MinHash/SimHash de-duplicator. */
export function dedupe(chunks: Chunk[], threshold = 0.8): Chunk[] {
  const words = (s: string) =>
    new Set(s.toLowerCase().split(/\W+/).filter(Boolean));
  const kept: { c: Chunk; w: Set<string> }[] = [];
  for (const c of [...chunks].sort((a, b) => b.score - a.score)) {
    const w = words(c.text);
    const dup = kept.some(({ w: kw }) => jaccard(w, kw) >= threshold);
    if (!dup) kept.push({ c, w });
  }
  return kept.map((k) => k.c);
}

function jaccard(a: Set<string>, b: Set<string>): number {
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

/** 1e. Source diversity cap — never let one file eat the whole window.
 *  Ten chunks from one file answer one question well and every other question
 *  badly. Capping per source forces breadth, which matters most for
 *  "how does X interact with Y" questions spanning several files. */
export function capPerSource(chunks: Chunk[], maxPerSource: number): Chunk[] {
  const seen = new Map<string, number>();
  const out: Chunk[] = [];
  for (const c of [...chunks].sort((a, b) => b.score - a.score)) {
    const n = seen.get(c.source) ?? 0;
    if (n >= maxPerSource) continue;
    seen.set(c.source, n + 1);
    out.push(c);
  }
  return out;
}

/** 1f. Tiered packing — full text for the top tier, compressed for the tail.
 *  Precision where it matters, recall where it's cheap: the best `fullTier`
 *  chunks go in verbatim (so they can be quoted exactly), everything after is
 *  squeezed through `compress` until the budget is exhausted. */
export function packTiered(
  chunks: Chunk[],
  budget: number,
  count: TokenCounter,
  fullTier: number,
  compress: (c: Chunk) => string
): Chunk[] {
  const ranked = [...chunks].sort((a, b) => b.score - a.score);
  const out: Chunk[] = [];
  let used = 0;
  ranked.forEach((c, i) => {
    const text = i < fullTier ? c.text : compress(c);
    const t = count(text);
    if (used + t > budget) return;
    out.push({ ...c, text });
    used += t;
  });
  return out;
}

/** Total tokens a chunk list will occupy (text only — formatting adds more). */
export function totalTokens(chunks: Chunk[], count: TokenCounter): number {
  return chunks.reduce((n, c) => n + count(c.text), 0);
}

// ===========================================================================
// 2. CONTEXT ORDERING — highest relevance at top and bottom, not the middle
// ===========================================================================
// "Lost in the middle" (Liu et al., 2023): with the same evidence in the
// window, accuracy is highest when the answer sits at the START or the END of
// the context and drops sharply when it is buried in the middle — a U-shaped
// curve. So the order you emit chunks in is a retrieval decision, not a
// cosmetic one, and simply sorting by score descending wastes the strong tail
// position on your weakest evidence.

/** 2a. Relevance descending — the naive baseline. Best evidence gets the
 *  strong head position; worst evidence gets the *other* strong position. */
export function orderByRelevance(chunks: Chunk[]): Chunk[] {
  return [...chunks].sort((a, b) => b.score - a.score);
}

/** 2b. Bookend / "V" ordering — the lost-in-the-middle fix.
 *  Alternate outwards-in: rank 1 first, rank 2 last, rank 3 second, rank 4
 *  second-to-last… so the strongest chunks occupy both attention peaks and the
 *  weakest sink into the middle where attention is cheapest anyway.
 *  This is the default you want for anything above ~5 chunks. */
export function orderBookend(chunks: Chunk[]): Chunk[] {
  const ranked = orderByRelevance(chunks);
  const head: Chunk[] = [];
  const tail: Chunk[] = [];
  ranked.forEach((c, i) => (i % 2 === 0 ? head.push(c) : tail.unshift(c)));
  return [...head, ...tail];
}

/** 2c. Document order restore — group chunks by source and put them back in
 *  reading order inside each group. Interleaved fragments of three files read
 *  as noise; contiguous runs read as a document. Sources are ordered by their
 *  best chunk, so relevance still drives the top-level layout.
 *  Use this when chunks are narrative/sequential (code files, tutorials). */
export function orderByDocument(chunks: Chunk[]): Chunk[] {
  const groups = new Map<string, Chunk[]>();
  for (const c of chunks) {
    const g = groups.get(c.source) ?? [];
    g.push(c);
    groups.set(c.source, g);
  }
  return [...groups.values()]
    .map((g) => g.sort((a, b) => (a.ordinal ?? 0) - (b.ordinal ?? 0)))
    .sort((a, b) => Math.max(...b.map((c) => c.score)) - Math.max(...a.map((c) => c.score)))
    .flat();
}

/** 2d. Group-then-bookend — the production compromise. Keep each source's
 *  chunks contiguous (coherence) but place whole *groups* at the attention
 *  peaks (recency/primacy). Best of 2b and 2c. */
export function orderGroupedBookend(chunks: Chunk[]): Chunk[] {
  const groups = new Map<string, Chunk[]>();
  for (const c of orderByDocument(chunks)) {
    const g = groups.get(c.source) ?? [];
    g.push(c);
    groups.set(c.source, g);
  }
  const ranked = [...groups.values()].sort(
    (a, b) => Math.max(...b.map((c) => c.score)) - Math.max(...a.map((c) => c.score))
  );
  const head: Chunk[][] = [];
  const tail: Chunk[][] = [];
  ranked.forEach((g, i) => (i % 2 === 0 ? head.push(g) : tail.unshift(g)));
  return [...head, ...tail].flat();
}

/** 2e. Query-adjacent placement — where the *question* goes matters as much
 *  as where the evidence goes. Repeating the query immediately after the
 *  context (so it occupies the final, strongest position) reliably beats
 *  asking once before a long context.
 *  Returns the three prompt parts in emission order. */
export function queryAdjacentLayout(
  query: string,
  context: string,
  instructions: string
): { pre: string; context: string; post: string } {
  return {
    pre: `${instructions}\n\nQuestion: ${query}`,
    context,
    post: `Question (restated): ${query}\nAnswer using only the context above, with citations.`,
  };
}

// ===========================================================================
// 3. CONTEXT COMPRESSION — summarising or trimming chunks before injection
// ===========================================================================
// Compression trades fidelity for budget. The rule that keeps it safe: never
// compress the evidence you expect to quote. Compress the supporting cast so
// the star witness fits verbatim.

const SENTENCE_SPLIT = /(?<=[.!?;])\s+/;

/** 3a. Extractive sentence selection — keep only the sentences of a chunk that
 *  overlap the query. Free, lossless at the sentence level, and it preserves
 *  the exact wording, so the result is still quotable. Start here. */
export function extractiveCompress(
  text: string,
  query: string,
  maxSentences: number
): string {
  const q = new Set(query.toLowerCase().split(/\W+/).filter(Boolean));
  const sentences = text.split(SENTENCE_SPLIT).filter(Boolean);
  return sentences
    .map((s, i) => ({
      s,
      i,
      score: score(s, q),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences)
    .sort((a, b) => a.i - b.i) // restore reading order
    .map((x) => x.s)
    .join(" ");
}

function score(sentence: string, q: Set<string>): number {
  const w = sentence.toLowerCase().split(/\W+/).filter(Boolean);
  if (w.length === 0) return 0;
  let hits = 0;
  for (const x of w) if (q.has(x)) hits++;
  return hits / Math.sqrt(w.length); // length-normalised overlap
}

/** 3b. Sentence-boundary truncation — hard cap that never cuts mid-sentence.
 *  A chunk severed mid-clause invites the model to complete the thought
 *  itself, which is exactly how truncation turns into hallucination. */
export function truncateToTokens(
  text: string,
  maxTokens: number,
  count: TokenCounter
): string {
  if (count(text) <= maxTokens) return text;
  const out: string[] = [];
  let used = 0;
  for (const s of text.split(SENTENCE_SPLIT)) {
    const t = count(s);
    if (used + t > maxTokens) break;
    out.push(s);
    used += t;
  }
  return out.length ? out.join(" ") + " […]" : text.slice(0, maxTokens * 4) + " […]";
}

/** 3c. Cross-chunk redundancy pruning — dedupe at *sentence* level across the
 *  whole context. Near-duplicate chunks survive 1d (they differ overall) while
 *  still repeating the same key sentence three times; this catches that. */
export function pruneRedundantSentences(chunks: Chunk[]): Chunk[] {
  const seen = new Set<string>();
  const norm = (s: string) => s.toLowerCase().replace(/\W+/g, " ").trim();
  return chunks.map((c) => {
    const kept = c.text.split(SENTENCE_SPLIT).filter((s) => {
      const k = norm(s);
      if (!k || seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    return { ...c, text: kept.join(" ") };
  }).filter((c) => c.text.trim().length > 0);
}

/** 3d. Abstractive (LLM) compression — a query-focused summary of a chunk.
 *  Highest compression ratio, highest cost, and the only lossy option: the
 *  output is paraphrase, so it can no longer be quoted verbatim. Mark such
 *  chunks as summaries in the prompt so the model doesn't invent quotes.
 *  `summarize` is injected — mock it in tests, point it at Haiku in prod. */
export function abstractiveCompress(
  chunk: Chunk,
  query: string,
  summarize: (text: string, query: string) => string
): Chunk {
  return { ...chunk, text: summarize(chunk.text, query), id: chunk.id };
}

/** 3e. Compression cascade — only compress when you must, cheapest first.
 *  Stage 0 dedupe → stage 1 sentence pruning → stage 2 extractive on the tail
 *  → stage 3 LLM summaries on the tail. Exits the moment the list fits, so a
 *  query whose evidence already fits pays nothing. */
export function compressionCascade(
  chunks: Chunk[],
  query: string,
  budget: number,
  count: TokenCounter,
  deps: {
    summarize: (text: string, query: string) => string;
    protectTop?: number; // never compress these — they get quoted
  }
): { chunks: Chunk[]; stages: string[]; summarised: Set<string> } {
  const protect = deps.protectTop ?? 1;
  const stages: string[] = [];
  const summarised = new Set<string>();
  let cur = orderByRelevance(chunks);
  const fits = () => totalTokens(cur, count) <= budget;

  stages.push(`start: ${totalTokens(cur, count)} tok vs budget ${budget}`);
  if (fits()) return { chunks: cur, stages: [...stages, "fits — no compression"], summarised };

  cur = dedupe(cur, 0.8);
  stages.push(`dedupe → ${totalTokens(cur, count)} tok`);
  if (fits()) return { chunks: cur, stages, summarised };

  cur = pruneRedundantSentences(cur);
  stages.push(`prune redundant sentences → ${totalTokens(cur, count)} tok`);
  if (fits()) return { chunks: cur, stages, summarised };

  cur = cur.map((c, i) =>
    i < protect ? c : { ...c, text: extractiveCompress(c.text, query, 2) }
  );
  stages.push(`extractive (tail) → ${totalTokens(cur, count)} tok`);
  if (fits()) return { chunks: cur, stages, summarised };

  cur = cur.map((c, i) => {
    if (i < protect) return c;
    summarised.add(c.id);
    return abstractiveCompress(c, query, deps.summarize);
  });
  stages.push(`LLM summaries (tail) → ${totalTokens(cur, count)} tok`);
  if (fits()) return { chunks: cur, stages, summarised };

  cur = packGreedy(cur, budget, count);
  stages.push(`still over — drop lowest-value chunks → ${totalTokens(cur, count)} tok`);
  return { chunks: cur, stages, summarised };
}

// ===========================================================================
// 4. CITATION-READY FORMATTING (sets up 3.9)
// ===========================================================================
// A model can only cite what you gave it a *handle* for. Formatting decides
// whether "according to the docs" or "[3] auth/session.go:L20-L48" is even
// expressible. Three requirements:
//   (i)   a stable, short marker per block ([1], [2], …)
//   (ii)  provenance next to the marker (source + location), so the answer's
//         citation can be resolved back to a file/line without a lookup table
//   (iii) hard delimiters, so the model can tell evidence from instructions —
//         and so injected text inside a chunk can't impersonate the prompt.

export interface FormattedContext {
  text: string; // the block to drop into the prompt
  /** marker → chunk, for verifying/rendering citations after generation. */
  citations: Map<number, Chunk>;
  tokens: number;
}

export interface FormatOptions {
  /** XML-ish tags fence evidence off from instructions; Claude follows them
   *  well and they survive chunks that themselves contain markdown/code. */
  style?: "xml" | "markdown";
  includeScores?: boolean; // useful for debugging, noise for the model
  /** Chunks rewritten by an LLM: paraphrase, so not quotable. */
  summaryIds?: Set<string>;
  /** Chunks trimmed extractively: shortened but still verbatim, so quotable. */
  excerptIds?: Set<string>;
}

export function formatForCitation(
  chunks: Chunk[],
  count: TokenCounter,
  opts: FormatOptions = {}
): FormattedContext {
  const style = opts.style ?? "xml";
  const citations = new Map<number, Chunk>();
  const blocks = chunks.map((c, i) => {
    const n = i + 1;
    citations.set(n, c);
    const loc = c.loc ? `:${c.loc}` : "";
    const kind = opts.summaryIds?.has(c.id)
      ? "summary" // paraphrased by an LLM — cite, don't quote
      : opts.excerptIds?.has(c.id)
        ? "excerpt" // sentences dropped, surviving text is verbatim
        : "";
    const kindAttr = kind ? ` kind="${kind}"` : "";
    const score = opts.includeScores ? ` score="${c.score.toFixed(3)}"` : "";
    if (style === "markdown") {
      return `[${n}] ${c.source}${loc}${kind ? ` (${kind})` : ""}\n${c.text}`;
    }
    return `<source id="${n}" path="${c.source}${loc}"${kindAttr}${score}>\n${c.text}\n</source>`;
  });
  const text =
    style === "xml"
      ? `<context>\n${blocks.join("\n")}\n</context>`
      : blocks.join("\n\n---\n\n");
  return { text, citations, tokens: count(text) };
}

/** The instruction half of the contract. Markers are worthless unless the
 *  system prompt tells the model exactly how to spend them — and tells it what
 *  to do when the context does not contain the answer (see 3.7). */
export function citationInstructions(): string {
  return [
    "Answer the question using ONLY the numbered sources in <context>.",
    "After every sentence that uses a source, cite it inline as [n] — e.g. “efSearch trades latency for recall [2].”",
    "Cite multiple sources as [1][3] when a sentence draws on several.",
    'Sources marked kind="excerpt" are shortened but verbatim — safe to quote.',
    'Sources marked kind="summary" are LLM paraphrases: cite them, but never quote them verbatim.',
    'If the context does not contain the answer, reply exactly: "I don\'t know based on the provided sources."',
    "Never cite a source number that does not appear in <context>.",
  ].join("\n");
}

/** Resolve the [n] markers a model emitted back to real chunks — the bridge
 *  into 3.9 Citations, where these get verified and rendered as links. */
export function resolveCitations(
  answer: string,
  citations: Map<number, Chunk>
): { used: Chunk[]; invalid: number[] } {
  const used: Chunk[] = [];
  const invalid: number[] = [];
  const seen = new Set<number>();
  for (const m of answer.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (seen.has(n)) continue;
    seen.add(n);
    const c = citations.get(n);
    if (c) used.push(c);
    else invalid.push(n); // hallucinated marker — 3.7's groundedness problem
  }
  return { used, invalid };
}

// ===========================================================================
// 5. THE COMPLETE CONTEXT CONSTRUCTION PIPELINE
// ===========================================================================
// Order of operations matters, and it is not the order the concepts are
// taught in:
//
//   dedupe → cap per source → compress to fit → pack under budget
//          → order for attention → format with citation markers
//
// Compress BEFORE packing (compression changes what fits), pack BEFORE
// ordering (ordering is meaningless on chunks that get dropped), and format
// LAST (formatting overhead must be inside the budget you measured).

export interface BuildConfig {
  contextWindow: number;
  systemTokens: number;
  answerReserve: number;
  maxPerSource: number;
  dedupeThreshold: number;
  protectTop: number; // top-N chunks never compressed (quotable evidence)
  ordering: "relevance" | "bookend" | "document" | "grouped-bookend";
  style?: "xml" | "markdown";
}

export interface BuildResult {
  prompt: string; // full prompt, ready to send
  context: FormattedContext;
  chunks: Chunk[]; // exactly what made it in, in emission order
  trace: string[];
}

export function buildContext(
  query: string,
  hits: Chunk[],
  cfg: BuildConfig,
  deps: {
    count: TokenCounter;
    summarize: (text: string, query: string) => string;
  }
): BuildResult {
  const { count } = deps;
  const trace: string[] = [];
  const instructions = citationInstructions();

  // --- Step 0: budget ------------------------------------------------------
  const budget = evidenceBudget({
    contextWindow: cfg.contextWindow,
    systemTokens: cfg.systemTokens + count(instructions),
    queryTokens: count(query) * 2, // query appears twice (2e)
    answerReserve: cfg.answerReserve,
  });
  trace.push(
    `budget: window=${cfg.contextWindow} − system/instructions − query×2 − answer=${cfg.answerReserve} ⇒ ${budget} tok for evidence`
  );

  // --- Step 1: cheap culling before anything expensive ---------------------
  let cur = dedupe(hits, cfg.dedupeThreshold);
  trace.push(`dedupe: ${hits.length} → ${cur.length} chunks`);
  cur = capPerSource(cur, cfg.maxPerSource);
  trace.push(`source cap (≤${cfg.maxPerSource}/source): ${cur.length} chunks, ${totalTokens(cur, count)} tok`);

  // --- Step 2: compress only if over budget --------------------------------
  const before = new Map(cur.map((c) => [c.id, c.text]));
  const { chunks: compressed, stages, summarised } = compressionCascade(cur, query, budget, count, {
    summarize: deps.summarize,
    protectTop: cfg.protectTop,
  });
  for (const s of stages) trace.push(`  compress: ${s}`);
  // Trimmed-but-verbatim vs paraphrased: the prompt must tell them apart, or
  // the model will "quote" a summary it invented the wording for.
  const excerptIds = new Set(
    compressed
      .filter((c) => before.get(c.id) !== c.text && !summarised.has(c.id))
      .map((c) => c.id)
  );

  // --- Step 3: pack (compression may still leave us over) ------------------
  const packed = packGreedy(compressed, budget, count);
  trace.push(`pack: ${packed.length}/${compressed.length} chunks kept, ${totalTokens(packed, count)}/${budget} tok used`);

  // --- Step 4: order for attention -----------------------------------------
  const ordered =
    cfg.ordering === "bookend"
      ? orderBookend(packed)
      : cfg.ordering === "document"
        ? orderByDocument(packed)
        : cfg.ordering === "grouped-bookend"
          ? orderGroupedBookend(packed)
          : orderByRelevance(packed);
  trace.push(`order (${cfg.ordering}): ${ordered.map((c) => c.id).join(" → ")}`);

  // --- Step 5: format with citation handles --------------------------------
  const context = formatForCitation(ordered, count, {
    style: cfg.style,
    summaryIds: summarised,
    excerptIds,
  });
  trace.push(`format: ${context.citations.size} citable sources, ${context.tokens} tok with markup`);

  // --- Step 6: assemble, query-adjacent ------------------------------------
  const layout = queryAdjacentLayout(query, context.text, instructions);
  const prompt = `${layout.pre}\n\n${layout.context}\n\n${layout.post}`;
  trace.push(`prompt: ${count(prompt)} tok total (answer reserve ${cfg.answerReserve} still free)`);

  return { prompt, context, chunks: ordered, trace };
}
