/**
 * ============================================================================
 *  QUERY REWRITING DEMO  —  run with:  npm start
 * ============================================================================
 *
 *  This file runs every strategy from ./rewriters.ts on the same sample
 *  question so you can compare their outputs side-by-side.
 *
 *  ALL LLM CALLS USE A MOCK — no API key or network required.
 *  The mock returns realistic hard-coded strings so the logic of every
 *  rewriter is visible. In production, replace `mockLLM` with a real
 *  Anthropic / OpenAI / Ollama client.
 *
 *  STRATEGY CHEAT SHEET
 *  --------------------
 *    Keyword Expansion ....... zero-latency synonym injection; narrow domains
 *    HyDE .................... embed a fake answer; good for dense retrieval
 *    Multi-Query ............. N paraphrases; union of results for higher recall
 *    Step-Back ............... add an abstract/background query alongside the original
 *    Sub-query Decomp. ....... split complex questions; retrieve per sub-question
 *    RAG-Fusion .............. multi-query + RRF re-ranking; best precision/recall balance
 *    Contextual Compress. .... condense chat history into one standalone query
 * ============================================================================
 */

import {
  keywordExpand,
  hydeRewrite,
  multiQueryRewrite,
  stepBackRewrite,
  subQueryDecompose,
  ragFusion,
  contextualCompress,
  LLMFn,
  RetrieverFn,
  ChatTurn,
  RewrittenQuery,
  FusedResult,
} from "./rewriters";

// ---------------------------------------------------------------------------
// SAMPLE DATA
// ---------------------------------------------------------------------------

const SAMPLE_QUERY =
  "How can I improve the retrieval speed of my database for large document collections?";

const SAMPLE_HISTORY: ChatTurn[] = [
  { role: "user", content: "What is PostgreSQL?" },
  {
    role: "assistant",
    content:
      "PostgreSQL is an open-source relational database known for ACID compliance and extensibility.",
  },
  { role: "user", content: "What about its scalability options?" },
];

// ---------------------------------------------------------------------------
// MOCK LLM
// ---------------------------------------------------------------------------
// Replace this with a real LLM call. Example using @anthropic-ai/sdk:
//
//   import Anthropic from "@anthropic-ai/sdk";
//   const client = new Anthropic();
//   const realLLM: LLMFn = async (prompt) => {
//     const msg = await client.messages.create({
//       model: "claude-haiku-4-5-20251001",
//       max_tokens: 512,
//       messages: [{ role: "user", content: prompt }],
//     });
//     return (msg.content[0] as { text: string }).text;
//   };

/**
 * A deterministic mock LLM that returns realistic responses based on keywords
 * in the prompt. Good enough to exercise every rewriting strategy without a
 * live API key.
 */
const mockLLM: LLMFn = async (prompt: string): Promise<string> => {
  const p = prompt.toLowerCase();

  // HyDE — return a plausible hypothetical answer passage.
  if (p.includes("factual passage") || p.includes("directly answers")) {
    return (
      "Retrieval speed for large document collections can be significantly improved " +
      "by adding vector indexes (e.g. HNSW or IVFFlat in pgvector), tuning the " +
      "similarity metric, partitioning the table, and caching frequent query " +
      "embeddings. Using approximate nearest-neighbour (ANN) search instead of " +
      "exact search trades a small recall drop for orders-of-magnitude speedup."
    );
  }

  // Multi-Query — return several paraphrases, one per line.
  if (p.includes("different phrasings") || p.includes("paraphrase")) {
    return [
      "What techniques speed up document search in large databases?",
      "How do I optimise vector search performance at scale?",
      "Best practices for fast retrieval over millions of text chunks?",
    ].join("\n");
  }

  // Step-Back — return a broader conceptual question.
  if (p.includes("more general question") || p.includes("step back")) {
    return "What are the key architectural factors that determine database retrieval performance?";
  }

  // Sub-query decomposition — return atomic sub-questions, one per line.
  if (p.includes("sub-questions") || p.includes("simpler")) {
    return [
      "What indexing strategies improve database retrieval speed?",
      "How does approximate nearest-neighbour search compare to exact search?",
      "What hardware configurations help with large-scale vector retrieval?",
      "How can query embedding caching reduce retrieval latency?",
    ].join("\n");
  }

  // Contextual compression — return a standalone question.
  if (p.includes("standalone question") || p.includes("conversation")) {
    return "What are the scalability options available in PostgreSQL for handling large workloads?";
  }

  // Fallback — echo the last sentence of the prompt as a trivial rewrite.
  const sentences = prompt.split(/[.!?]+/).map((s) => s.trim()).filter(Boolean);
  return sentences[sentences.length - 1] ?? prompt;
};

// ---------------------------------------------------------------------------
// MOCK RETRIEVER (for RAG-Fusion demo)
// ---------------------------------------------------------------------------
// Simulates a vector store that returns different ranked lists for different
// query strings. In production, replace with a real pgvector / Pinecone / Weaviate call.

const mockRetriever: RetrieverFn = async (query: string): Promise<Array<[string, number]>> => {
  // Each "query variant" retrieves a slightly different ordering of the same
  // pool of documents. This mirrors what real retrieval looks like.
  const pool: Record<string, Array<[string, number]>> = {
    default: [
      ["doc-index-tuning", 0.92],
      ["doc-ann-search", 0.88],
      ["doc-caching", 0.81],
      ["doc-hardware", 0.74],
      ["doc-partitioning", 0.65],
    ],
    speed: [
      ["doc-ann-search", 0.95],
      ["doc-index-tuning", 0.85],
      ["doc-hardware", 0.80],
      ["doc-caching", 0.70],
    ],
    scale: [
      ["doc-partitioning", 0.90],
      ["doc-index-tuning", 0.86],
      ["doc-caching", 0.78],
      ["doc-ann-search", 0.72],
    ],
    optimize: [
      ["doc-index-tuning", 0.91],
      ["doc-caching", 0.87],
      ["doc-ann-search", 0.82],
      ["doc-hardware", 0.68],
    ],
  };

  // Pick the result list that best matches a keyword in the query.
  const q = query.toLowerCase();
  if (q.includes("speed") || q.includes("fast")) return pool.speed;
  if (q.includes("scale") || q.includes("million")) return pool.scale;
  if (q.includes("optimis") || q.includes("optimiz")) return pool.optimize;
  return pool.default;
};

// ---------------------------------------------------------------------------
// DISPLAY HELPERS
// ---------------------------------------------------------------------------

const HR = "=".repeat(76);

function header(title: string): void {
  console.log(`\n${HR}\n${title}\n${HR}`);
}

function showQueries(queries: RewrittenQuery[]): void {
  queries.forEach((q, i) => {
    const meta = q.metadata ? `  ${JSON.stringify(q.metadata)}` : "";
    console.log(`  [${i}] "${q.query}"${meta}`);
  });
}

function showFused(results: FusedResult[]): void {
  results.forEach((r, i) => {
    console.log(
      `  [${i}] docId="${r.docId}"  rrfScore=${r.rrfScore.toFixed(5)}  appearsIn=${r.appearsIn}`
    );
  });
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log(`\nSample query:\n  "${SAMPLE_QUERY}"\n`);

  // 1. Keyword Expansion (no LLM)
  header("1. KEYWORD EXPANSION  (rule-based, zero LLM latency)");
  const expanded = keywordExpand(SAMPLE_QUERY, { appendMode: true });
  console.log(`  "${expanded.query}"`);

  // 2. HyDE
  header("2. HyDE — Hypothetical Document Embeddings");
  const hyde = await hydeRewrite(SAMPLE_QUERY, mockLLM, { maxWords: 80 });
  console.log(`  Hypothetical passage to embed:\n  "${hyde.query}"`);

  // 3. Multi-Query
  header("3. MULTI-QUERY  (numVariants=3)");
  const mq = await multiQueryRewrite(SAMPLE_QUERY, mockLLM, { numVariants: 3 });
  showQueries(mq);

  // 4. Step-Back
  header("4. STEP-BACK PROMPTING");
  const sb = await stepBackRewrite(SAMPLE_QUERY, mockLLM);
  showQueries(sb);

  // 5. Sub-query Decomposition
  header("5. SUB-QUERY DECOMPOSITION  (maxSubQueries=4)");
  const sq = await subQueryDecompose(SAMPLE_QUERY, mockLLM, { maxSubQueries: 4 });
  showQueries(sq);

  // 6. RAG-Fusion
  header("6. RAG-FUSION  (numQueries=4, k=60)");
  const fused = await ragFusion(SAMPLE_QUERY, mockLLM, mockRetriever, { numQueries: 4 });
  console.log("  Fused & re-ranked documents:");
  showFused(fused);

  // 7. Contextual Compression
  header("7. CONTEXTUAL COMPRESSION  (chat history → standalone query)");
  console.log("  Chat history:");
  SAMPLE_HISTORY.forEach((t) => console.log(`    ${t.role}: ${t.content}`));
  const compressed = await contextualCompress(SAMPLE_HISTORY, mockLLM);
  console.log(`\n  Standalone query:\n  "${compressed.query}"`);
  console.log(`  ${JSON.stringify(compressed.metadata)}`);

  console.log(`\n${HR}`);
  console.log("Done. Replace mockLLM with a real provider to use in production.");
  console.log(HR);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
