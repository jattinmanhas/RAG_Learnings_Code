import OpenAI from "openai";
import { client } from "./client.js";

// Tool (a.k.a. function) calling is the foundation of agentic RAG. You describe
// functions to the model; instead of answering directly, it can reply "call
// search_docs with { query: ... }". YOUR code runs the function and feeds the
// result back. The model then writes the final answer grounded in that result.
//
// The loop:
//   1. Send messages + tool definitions.
//   2. If the model returns tool_calls, execute each locally.
//   3. Append results as role:"tool" messages and call again.
//   4. Repeat until the model answers with plain content.

// A local "tool" — here a fake document search standing in for your retriever.
function searchDocs(query: string): string {
  return `Top result for "${query}": pgvector supports HNSW and IVFFlat indexes. \
HNSW gives higher recall and faster queries at the cost of slower build time and \
more memory; IVFFlat is cheaper to build.`;
}

const tools: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_docs",
      description: "Search the knowledge base for passages relevant to a query.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "The search query." },
        },
        required: ["query"],
      },
    },
  },
];

const messages: OpenAI.ChatCompletionMessageParam[] = [
  {
    role: "system",
    content:
      "You answer using the search_docs tool when you need facts. Cite what you find.",
  },
  {
    role: "user",
    content: "Which pgvector index gives better recall, HNSW or IVFFlat?",
  },
];

// Allow a few rounds so the model can search, then answer.
for (let round = 0; round < 4; round++) {
  const response = await client.chat.completions.create({
    model: "llama-3.3-70b-versatile", // tool use wants a stronger model
    messages,
    tools,
  });

  const msg = response.choices[0].message;

  // No tool calls → final answer.
  if (!msg.tool_calls || msg.tool_calls.length === 0) {
    console.log(msg.content);
    break;
  }

  messages.push(msg); // echo the assistant turn (with tool_calls) back

  for (const call of msg.tool_calls) {
    if (call.type !== "function") continue;
    const args = JSON.parse(call.function.arguments) as { query: string };

    const result =
      call.function.name === "search_docs"
        ? searchDocs(args.query)
        : `unknown tool "${call.function.name}"`;

    console.log(`  [called ${call.function.name}("${args.query}")]`);
    messages.push({
      role: "tool",
      tool_call_id: call.id,
      content: result,
    });
  }
}
