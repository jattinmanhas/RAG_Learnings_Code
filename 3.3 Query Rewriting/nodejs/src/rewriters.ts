/**
 * ============================================================================
 *  QUERY REWRITING STRATEGIES  (Node.js / TypeScript)
 * ============================================================================
 *
 *  WHY REWRITE THE QUERY AT ALL?
 *  -----------------------------
 *  In a RAG pipeline the retrieval step lives or dies on query quality.
 *  Users write short, ambiguous, or conversational questions that don't
 *  match the way information is stored in your vector index.
 *
 *  Example problem:
 *    User asks: "how do I make it go faster?"
 *    Stored chunk: "Performance tuning in PostgreSQL can be achieved by…"
 *    → Embedding distance is large because the vocabularies barely overlap.
 *
 *  Query rewriting bridges that gap BEFORE you ever hit the vector store.
 *  You transform the raw question into one or more forms that retrieve
 *  better documents, then generate the final answer from those documents.
 *
 *  THE MAIN STRATEGIES COVERED HERE
 *  ---------------------------------
 *    1.  Keyword Expansion    — rule-based synonym injection (no LLM)
 *    2.  HyDE                 — generate a hypothetical answer; embed that
 *    3.  Multi-Query          — N paraphrase variants; union of results
 *    4.  Step-Back Prompting  — rephrase to a broader, more abstract question
 *    5.  Sub-query Decomp.    — split into atomic sub-questions; merge results
 *    6.  RAG-Fusion           — multi-query + Reciprocal Rank Fusion
 *    7.  Contextual Compress. — condense a multi-turn conversation into one
 *                               standalone question
 *
 *  ARCHITECTURE NOTE
 *  -----------------
 *  All LLM-dependent strategies accept an `LLMFn` callback:
 *
 *      type LLMFn = (prompt: string) => Promise<string>
 *
 *  This keeps every function dependency-free — you plug in your own client
 *  (Anthropic, OpenAI, Ollama, …) or a mock for tests.
 *  See index.ts for an example mock.
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

/**
 * A function that calls an LLM with `prompt` and returns the text response.
 * Implement this with your real provider (Anthropic, OpenAI, etc.).
 */
export type LLMFn = (prompt: string) => Promise<string>;

/**
 * A function that retrieves ranked document IDs for a given query.
 * Used by the RAG-Fusion strategy to demonstrate re-ranking.
 *
 * Returns an array of [docId, score] pairs, highest score first.
 */
export type RetrieverFn = (query: string) => Promise<Array<[string, number]>>;

/** A rewritten query plus the strategy that produced it. */
export interface RewrittenQuery {
  /** The rewritten (or expanded) query string. */
  query: string;
  /** Human-readable label of the strategy used. */
  strategy: string;
  /** Optional extra metadata (e.g. which sub-question this is). */
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// HELPER
// ---------------------------------------------------------------------------

/**
 * Split an LLM response that lists items one per line into a clean array.
 * Strips leading numbers, bullets, dashes, and surrounding whitespace.
 */
function splitLines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.replace(/^\s*[-\d.)+*]+\s*/, "").trim())
    .filter((l) => l.length > 0);
}

/* ===========================================================================
 *  1. KEYWORD EXPANSION  (rule-based — no LLM)
 * ===========================================================================
 *
 *  WHAT IT IS
 *      Add synonyms or domain-specific alternate terms to the raw query so
 *      the embedding covers more of the semantic neighbourhood.
 *
 *  HOW IT WORKS
 *      We maintain a hand-crafted synonym map (or pull one from a thesaurus /
 *      ontology). For each word in the query that appears in the map, we
 *      append its synonyms.
 *
 *  WHEN TO USE
 *      - You have a known, narrow domain (legal, medical, finance) with stable
 *        terminology.
 *      - You need zero latency — no round-trip to an LLM.
 *      - Good as a pre-filter before any of the LLM strategies below.
 *
 *  DOWNSIDE
 *      The synonym map must be maintained manually. It won't catch novel or
 *      contextual meanings ("bank" = riverbank vs. financial institution).
 * ===========================================================================
 */
export interface KeywordExpansionOptions {
  /** Custom synonym map. Each key maps to a list of alternate terms. */
  synonymMap?: Record<string, string[]>;
  /** If true, append synonyms as extra context rather than replacing words. */
  appendMode?: boolean;
}

/** Built-in general-purpose synonym map for RAG-domain queries. */
const DEFAULT_SYNONYMS: Record<string, string[]> = {
  fast: ["quick", "rapid", "performant", "speedy"],
  slow: ["sluggish", "latent", "low-throughput"],
  error: ["bug", "exception", "failure", "fault", "issue"],
  improve: ["enhance", "optimize", "boost", "increase"],
  database: ["db", "datastore", "storage", "repository"],
  retrieve: ["fetch", "get", "query", "search", "find"],
  document: ["doc", "file", "record", "text", "passage"],
  summarize: ["summarise", "condense", "abstract", "overview"],
};

/**
 * Expand the query by injecting synonyms for recognised keywords.
 *
 * @example
 * keywordExpand("how to improve database retrieval speed")
 * // → "how to improve|enhance|optimize|boost database|db retrieval|fetch speed|fast|quick"
 */
export function keywordExpand(
  query: string,
  options: KeywordExpansionOptions = {}
): RewrittenQuery {
  const map = { ...DEFAULT_SYNONYMS, ...(options.synonymMap ?? {}) };
  const words = query.split(/\s+/);

  const expanded = words.map((word) => {
    const key = word.toLowerCase().replace(/[^a-z]/g, "");
    const syns = map[key];
    if (!syns) return word;
    // Append mode: keep original word, add synonyms as alternatives.
    return options.appendMode ? `${word} ${syns.join(" ")}` : [word, ...syns].join("|");
  });

  return {
    query: expanded.join(" "),
    strategy: "keyword-expansion",
    metadata: { originalQuery: query },
  };
}

/* ===========================================================================
 *  2. HyDE — Hypothetical Document Embeddings
 * ===========================================================================
 *
 *  WHAT IT IS
 *      Instead of embedding the user's question, ask the LLM to write a
 *      SHORT HYPOTHETICAL ANSWER, then embed THAT. The fake answer lives in
 *      the same vector space as real documents, so the nearest-neighbour
 *      search retrieves documents similar to "what the answer looks like"
 *      rather than "what the question looks like".
 *
 *  HOW IT WORKS
 *      1. Prompt the LLM: "Write a short passage that answers: {query}"
 *      2. Embed the returned passage (not the query).
 *      3. Use that embedding as the retrieval vector.
 *
 *  WHEN TO USE
 *      - Short, keyword-heavy queries that don't resemble documents.
 *      - Dense retrieval where the gap between question and answer style is
 *        large (e.g. "who invented X?" vs. a Wikipedia paragraph about X).
 *
 *  DOWNSIDE
 *      Adds one LLM round-trip per query. The hallucinated answer may drift
 *      from reality — you're betting that even wrong details produce an
 *      embedding closer to the right documents than the bare question does.
 *
 *  REFERENCE  Gao et al., 2022  "Precise Zero-Shot Dense Retrieval without
 *             Relevance Labels"  (arxiv 2212.10496)
 * ===========================================================================
 */
export interface HyDEOptions {
  /** Maximum words the LLM should write for the hypothetical passage. */
  maxWords?: number;
}

/**
 * Generate a hypothetical answer document for the query.
 * Embed the returned `.query` string instead of the original query.
 */
export async function hydeRewrite(
  query: string,
  llm: LLMFn,
  options: HyDEOptions = {}
): Promise<RewrittenQuery> {
  const maxWords = options.maxWords ?? 120;

  const prompt = `Write a factual passage of at most ${maxWords} words that directly \
answers the following question. Do NOT include the question itself in your response.

Question: ${query}

Passage:`;

  const hypothetical = (await llm(prompt)).trim();

  return {
    // We return the hypothetical answer as the "query" — the caller embeds this.
    query: hypothetical,
    strategy: "hyde",
    metadata: { originalQuery: query },
  };
}

/* ===========================================================================
 *  3. MULTI-QUERY (Query Expansion via LLM)
 * ===========================================================================
 *
 *  WHAT IT IS
 *      Ask the LLM to rephrase the original question N different ways.
 *      Run retrieval for EACH variant, then take the union (or weighted
 *      union) of all returned documents.
 *
 *  HOW IT WORKS
 *      1. Prompt: "List {n} different ways to phrase: {query}"
 *      2. Parse the N variants.
 *      3. Retrieve documents for each. Union the result sets.
 *
 *  WHEN TO USE
 *      - The query is underspecified or could be interpreted several ways.
 *      - You want to increase recall without the cost of a full re-ranker.
 *
 *  DOWNSIDE
 *      N × retrieval latency. Deduplication of results adds complexity.
 *      With a naïve union, less-relevant documents from weak variants can
 *      dilute the final context. Use RAG-Fusion (strategy 6) to fix that.
 *
 *  NOTE
 *      The original query is always included as the first variant — that way
 *      you never lose the exact intent.
 * ===========================================================================
 */
export interface MultiQueryOptions {
  /** How many paraphrase variants to generate (default: 3). */
  numVariants?: number;
}

/**
 * Generate multiple paraphrases of the query.
 * Run retrieval for each and union the results.
 */
export async function multiQueryRewrite(
  query: string,
  llm: LLMFn,
  options: MultiQueryOptions = {}
): Promise<RewrittenQuery[]> {
  const n = options.numVariants ?? 3;

  const prompt = `Generate ${n} different phrasings of the following question. \
Each phrasing should preserve the original intent but use different words or structure. \
Output ONLY the ${n} questions, one per line, no numbering or bullets.

Original question: ${query}`;

  const raw = await llm(prompt);
  const variants = splitLines(raw).slice(0, n);

  // Always include the original query as the first entry.
  return [
    { query, strategy: "multi-query", metadata: { variant: 0, isOriginal: true } },
    ...variants.map((q, i) => ({
      query: q,
      strategy: "multi-query",
      metadata: { variant: i + 1, isOriginal: false },
    })),
  ];
}

/* ===========================================================================
 *  4. STEP-BACK PROMPTING
 * ===========================================================================
 *
 *  WHAT IT IS
 *      Rephrase a specific, detailed question into a more ABSTRACT question
 *      that retrieves background / foundational knowledge. Then use BOTH the
 *      original and the step-back query to retrieve documents.
 *
 *  HOW IT WORKS
 *      1. Prompt: "What is a more general question behind: {query}?"
 *      2. Retrieve documents for BOTH queries (union).
 *      3. The step-back results supply context; the original results supply
 *         specifics.
 *
 *  EXAMPLE
 *      Original : "What was the temperature in Miami on 1 Jan 2022?"
 *      Step-back: "What factors determine temperatures in subtropical cities?"
 *      → The step-back retrieves climatology articles that explain WHY
 *        the temperature was what it was.
 *
 *  WHEN TO USE
 *      - Questions that require background knowledge to be answerable.
 *      - Science / engineering / legal questions with implicit prerequisites.
 *
 *  REFERENCE  Zheng et al., 2023  "Take a Step Back: Evoking Reasoning in
 *             LLMs via Abstraction"  (arxiv 2310.06117)
 * ===========================================================================
 */

/**
 * Generate an abstract "step-back" question that retrieves background context.
 * Run retrieval for BOTH this and the original, then merge.
 */
export async function stepBackRewrite(
  query: string,
  llm: LLMFn
): Promise<RewrittenQuery[]> {
  const prompt = `Given the following specific question, write ONE more general question \
that captures the underlying concept or background knowledge required to answer it. \
Output ONLY the general question, nothing else.

Specific question: ${query}

General question:`;

  const stepBack = (await llm(prompt)).trim();

  return [
    // Always keep the original for specific-fact retrieval.
    { query, strategy: "step-back", metadata: { role: "specific" } },
    // The step-back query retrieves foundational/background knowledge.
    { query: stepBack, strategy: "step-back", metadata: { role: "abstract" } },
  ];
}

/* ===========================================================================
 *  5. SUB-QUERY DECOMPOSITION
 * ===========================================================================
 *
 *  WHAT IT IS
 *      Break a COMPLEX, multi-faceted question into SIMPLER atomic sub-
 *      questions, retrieve documents for each, then synthesise an answer
 *      from the combined evidence.
 *
 *  HOW IT WORKS
 *      1. Prompt: "Break this into sub-questions: {query}"
 *      2. Parse each sub-question.
 *      3. Retrieve documents per sub-question.
 *      4. Feed all documents (labelled by sub-question) to the generator.
 *
 *  EXAMPLE
 *      Original  : "Compare the performance and scalability of PostgreSQL vs
 *                   MongoDB for a read-heavy e-commerce workload."
 *      Sub-queries: 1. "PostgreSQL performance characteristics for reads"
 *                   2. "MongoDB performance characteristics for reads"
 *                   3. "PostgreSQL scalability options"
 *                   4. "MongoDB scalability options"
 *                   5. "e-commerce read workload patterns"
 *
 *  WHEN TO USE
 *      - Comparative or multi-part questions.
 *      - Questions that require gathering evidence from different areas of
 *        the knowledge base before synthesis.
 *
 *  REFERENCE  Patel & Yang, 2024  "From Local to Global: A Graph RAG approach
 *             to Query-Focused Summarization"  (sub-question decomposition
 *             is a core component of Graph RAG pipelines)
 * ===========================================================================
 */
export interface SubQueryOptions {
  /** Maximum number of sub-questions to generate (default: 5). */
  maxSubQueries?: number;
}

/**
 * Decompose a complex query into simpler, independently-answerable sub-queries.
 */
export async function subQueryDecompose(
  query: string,
  llm: LLMFn,
  options: SubQueryOptions = {}
): Promise<RewrittenQuery[]> {
  const max = options.maxSubQueries ?? 5;

  const prompt = `Break the following complex question into at most ${max} simpler, \
self-contained sub-questions. Each sub-question should be independently answerable. \
Output ONLY the sub-questions, one per line, no numbering or bullets.

Complex question: ${query}`;

  const raw = await llm(prompt);
  const subQueries = splitLines(raw).slice(0, max);

  return subQueries.map((q, i) => ({
    query: q,
    strategy: "sub-query-decomposition",
    metadata: { subQueryIndex: i, totalSubQueries: subQueries.length },
  }));
}

/* ===========================================================================
 *  6. RAG-FUSION
 * ===========================================================================
 *
 *  WHAT IT IS
 *      Multi-Query (strategy 3) + RECIPROCAL RANK FUSION (RRF) to merge the
 *      N result lists into a single re-ranked list. RRF avoids the problem
 *      where naïve union over-promotes documents that appear in many weak
 *      result sets.
 *
 *  HOW RRF WORKS
 *      For each document d that appears in ANY result list:
 *
 *          RRF_score(d) = Σ  1 / (k + rank_i(d))
 *                          i
 *
 *      where rank_i(d) is the 1-based rank of d in list i (or ∞ if absent),
 *      and k is a smoothing constant (default 60, empirically best for most
 *      benchmarks).
 *
 *  HOW IT WORKS  (end-to-end)
 *      1. Generate N query variants (same as multi-query).
 *      2. Run retrieval for each — get ranked doc lists.
 *      3. Apply RRF across all lists.
 *      4. Return documents sorted by RRF score.
 *
 *  WHEN TO USE
 *      - You already run multi-query and want better precision.
 *      - You have multiple retrievers (BM25 + dense) whose results you want
 *        to fuse — RRF is the standard way to merge heterogeneous ranked lists.
 *
 *  REFERENCE  Cormack et al., 2009  "Reciprocal Rank Fusion outperforms
 *             Condorcet and individual rank learning methods"  (SIGIR 2009)
 *             + Shi et al., 2023  "RAG-Fusion: Improving LLM-based Retrieval
 *             Augmented Generation Systems"
 * ===========================================================================
 */
export interface RAGFusionOptions {
  /** Number of query variants to generate (default: 4). */
  numQueries?: number;
  /**
   * RRF smoothing constant. Higher k makes ranks influence scores less
   * dramatically. Default 60 is the standard recommendation.
   */
  k?: number;
}

export interface FusedResult {
  /** Document identifier (whatever your retriever returns). */
  docId: string;
  /** RRF score — higher is better. */
  rrfScore: number;
  /** How many query variants retrieved this document. */
  appearsIn: number;
}

/**
 * Generate multiple query variants, run a retriever for each, and fuse the
 * ranked lists with Reciprocal Rank Fusion.
 *
 * @param retriever  A function that takes a query and returns ranked [docId, score] pairs.
 * @returns           A fused, re-ranked array of {docId, rrfScore, appearsIn}.
 */
export async function ragFusion(
  query: string,
  llm: LLMFn,
  retriever: RetrieverFn,
  options: RAGFusionOptions = {}
): Promise<FusedResult[]> {
  const numQueries = options.numQueries ?? 4;
  const k = options.k ?? 60;

  // Step 1 — Generate query variants via multi-query rewriting.
  const variants = await multiQueryRewrite(query, llm, { numVariants: numQueries - 1 });
  const queries = variants.map((v) => v.query);

  // Step 2 — Retrieve documents for each variant in parallel.
  const allRankedLists = await Promise.all(queries.map((q) => retriever(q)));

  // Step 3 — Reciprocal Rank Fusion across all ranked lists.
  //
  // rrfScores maps docId → accumulated RRF score.
  // appearsInCount tracks in how many lists a doc appears (useful for tie-breaking).
  const rrfScores = new Map<string, number>();
  const appearsInCount = new Map<string, number>();

  for (const rankedList of allRankedLists) {
    rankedList.forEach(([docId], zeroBasedIdx) => {
      const rank = zeroBasedIdx + 1; // RRF uses 1-based rank
      const contribution = 1 / (k + rank);
      rrfScores.set(docId, (rrfScores.get(docId) ?? 0) + contribution);
      appearsInCount.set(docId, (appearsInCount.get(docId) ?? 0) + 1);
    });
  }

  // Step 4 — Sort by descending RRF score and return.
  return Array.from(rrfScores.entries())
    .map(([docId, rrfScore]) => ({
      docId,
      rrfScore,
      appearsIn: appearsInCount.get(docId) ?? 0,
    }))
    .sort((a, b) => b.rrfScore - a.rrfScore);
}

/* ===========================================================================
 *  7. CONTEXTUAL QUERY COMPRESSION (Conversation History Condensation)
 * ===========================================================================
 *
 *  WHAT IT IS
 *      In a multi-turn chat, the user's latest message may reference earlier
 *      turns: "What about its scalability?" — "it" could be PostgreSQL from
 *      three messages ago. The retriever has no memory of prior turns, so
 *      this pronoun-heavy question returns garbage.
 *
 *      Contextual compression rewrites the latest message into a STANDALONE
 *      question that a fresh retriever can understand with no context.
 *
 *  HOW IT WORKS
 *      1. Format the conversation history as a transcript.
 *      2. Prompt: "Rewrite the last question as a standalone query."
 *      3. Use the rewritten query for retrieval; keep the history for the
 *         generator's context window.
 *
 *  WHEN TO USE
 *      - Any conversational / chat-over-documents interface.
 *      - ALWAYS apply this before retrieval in multi-turn RAG — the cost is
 *        trivial and the precision gain is large.
 *
 *  REFERENCE  LangChain "Contextual Compression Retriever" pattern;
 *             popularised by the ConversationalRetrievalChain abstraction.
 * ===========================================================================
 */

/** A single turn in a conversation. */
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/**
 * Compress a multi-turn conversation into a single, standalone retrieval query.
 *
 * @param history  The conversation so far. The LAST user message is the one
 *                 being rewritten; all preceding turns provide context.
 */
export async function contextualCompress(
  history: ChatTurn[],
  llm: LLMFn
): Promise<RewrittenQuery> {
  if (history.length === 0) {
    throw new Error("contextualCompress: history must have at least one turn.");
  }

  // The question we want to rewrite is the last user message.
  const lastUserTurn = [...history].reverse().find((t) => t.role === "user");
  if (!lastUserTurn) throw new Error("contextualCompress: no user turn found in history.");

  // Format the full chat history for the LLM.
  const transcript = history
    .map((t) => `${t.role === "user" ? "User" : "Assistant"}: ${t.content}`)
    .join("\n");

  const prompt = `Given the following conversation, rewrite the final user question \
as a STANDALONE question that contains all the context needed for a search engine to \
retrieve relevant documents — with no references to "it", "they", "the above", etc.
Output ONLY the rewritten question, nothing else.

Conversation:
${transcript}

Standalone question:`;

  const rewritten = (await llm(prompt)).trim();

  return {
    query: rewritten,
    strategy: "contextual-compression",
    metadata: {
      originalQuestion: lastUserTurn.content,
      historyLength: history.length,
    },
  };
}
