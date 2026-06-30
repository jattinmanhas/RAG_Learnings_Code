import { client } from "./client.js";

// For RAG you rarely want prose — you want DATA your code can act on:
// a parsed entity, a routing decision, a relevance score. Asking nicely for
// JSON is unreliable (the model adds prose, code fences, apologies).
// response_format: { type: "json_object" } forces ONLY valid JSON.
//
// Two rules when using JSON mode:
//   1. The prompt MUST contain the word "json" (the API enforces this).
//   2. JSON mode guarantees valid JSON, not YOUR schema — describe the exact
//      shape you want in the prompt, then validate after parsing.

interface QueryAnalysis {
  intent: "factual" | "summarization" | "comparison" | "chitchat";
  entities: string[];
  needs_rag: boolean; // false for greetings / chitchat
  rewrite: string;    // keyword-rich query for vector search
}

async function analyzeQuery(userQuery: string): Promise<QueryAnalysis> {
  const response = await client.chat.completions.create({
    model: "llama-3.1-8b-instant",
    response_format: { type: "json_object" },
    temperature: 0, // deterministic structured output
    max_tokens: 512,
    messages: [
      {
        role: "user",
        content: `Analyze the user's search query and respond with a JSON object
containing exactly these fields:
  "intent":    one of "factual", "summarization", "comparison", "chitchat"
  "entities":  array of key named entities in the query
  "needs_rag": boolean, false for greetings/chitchat
  "rewrite":   a concise, keyword-rich version of the query for vector search

User query: ${JSON.stringify(userQuery)}`,
      },
    ],
  });

  const raw = response.choices[0].message.content ?? "{}";
  return JSON.parse(raw) as QueryAnalysis; // wrap in try/catch in production
}

const analysis = await analyzeQuery(
  "how does pgvector's HNSW index compare to IVFFlat for recall?"
);
console.log(analysis);
