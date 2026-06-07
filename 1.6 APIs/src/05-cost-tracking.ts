import { client } from "./client.js";

interface RequestCost {
  model: string;
  inputTokens: number;
  outputTokens: number;
  costUSD: number;
  durationMs: number;
}

// Price per 1M tokens (approximate, check current pricing)
const PRICING: Record<string, { input: number; output: number }> = {
  "gpt-4o":              { input: 2.50,  output: 10.00 },
  "gpt-4o-mini":         { input: 0.15,  output: 0.60  },
  "claude-sonnet-4":     { input: 3.00,  output: 15.00 },
  "llama-3.1-8b-instant":{ input: 0.05,  output: 0.08  }, // Groq pricing
};

async function trackedCall(
  prompt: string,
  model = "llama-3.1-8b-instant"
): Promise<{ text: string; cost: RequestCost }> {
  
  const start = Date.now();

  const response = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: prompt }],
    max_tokens: 512,
  });

  const durationMs = Date.now() - start;
  const usage = response.usage!;
  const prices = PRICING[model] ?? { input: 0, output: 0 };

  const costUSD =
    (usage.prompt_tokens     / 1_000_000) * prices.input +
    (usage.completion_tokens / 1_000_000) * prices.output;

  const cost: RequestCost = {
    model,
    inputTokens:  usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    costUSD,
    durationMs,
  };

  console.log(`
    Model:   ${model}
    Tokens:  ${usage.prompt_tokens} in / ${usage.completion_tokens} out
    Cost:    $${costUSD.toFixed(6)}
    Time:    ${durationMs}ms
  `);

  return {
    text: response.choices[0].message.content ?? "",
    cost,
  };
}


// Accumulate cost across a session
let sessionCost = 0;

const r1 = await trackedCall("What is RAG?");
sessionCost += r1.cost.costUSD;

const r2 = await trackedCall("What is pgvector?");
sessionCost += r2.cost.costUSD;

console.log(`Total session cost: $${sessionCost.toFixed(6)}`);
