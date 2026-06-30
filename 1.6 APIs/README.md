# 1.6 LLM APIs

Before you can build RAG, you need to talk to a language model reliably. This
section covers the API fundamentals every later stage depends on — basic calls,
conversation state, failure handling, streaming, cost, structured output, and
tool calling.

All examples talk to **Groq** through its **OpenAI-compatible** API. Groq is free
and fast, which is ideal for learning. Because the wire protocol is the OpenAI
standard, the same code points at OpenAI, Together, Fireworks, or a local Ollama
server by changing only the base URL, model name, and API key.

> Working with Anthropic's Claude instead? The concepts are identical — system
> prompt, message history, streaming, tokens, tool use — only the SDK surface
> and field names differ.

## Setup

Get a free key at [console.groq.com](https://console.groq.com), then create a
`.env` file in this folder (`1.6 APIs/.env`):

```
GROQ_API_KEY=your_key_here
```

Both the Node and Go examples read this same file.

## Available models (Groq, free)

| Model | Notes |
| --- | --- |
| `llama-3.1-8b-instant` | fast, good for learning (default in the examples) |
| `llama-3.3-70b-versatile` | smarter, slower — used for tool calling |
| `mixtral-8x7b-32768` | large context window |

## The examples

| # | Concept | Why it matters for RAG |
| --- | --- | --- |
| 01 | **Basic call** | one system prompt + one user turn → one answer |
| 02 | **Multi-turn conversation** | the API is stateless; you resend the full history every turn |
| 03 | **Retry & backoff** | survive 429/500/503; obey `Retry-After`; exponential backoff + jitter |
| 04 | **Streaming + silence timeout** | print tokens live, and abort if the stream stalls mid-answer |
| 05 | **Cost tracking** | read the `usage` block, price it, accumulate per session |
| 06 | **Structured output (JSON mode)** | force machine-readable JSON for query routing / classification |
| 07 | **Tool calling** | let the model call your retriever — the basis of agentic RAG |

### Key ideas

- **Stateless API (02).** The model remembers nothing. "Memory" is just you
  resending the prior messages each request — which is also why long chats cost
  more: input tokens grow every turn.
- **Retry only retryable errors (03).** A `400` (bad request) or `401` (bad key)
  will never succeed on retry — fail fast. Retry `429`/`500`/`503`/`408`, and
  when the server sends `Retry-After`, wait exactly that long.
- **Streaming needs its own timeout (04).** A stalled stream often does not error;
  it just goes quiet. Reset a watchdog on every token and abort on silence.
- **JSON mode guarantees valid JSON, not your schema (06).** Describe the exact
  fields in the prompt (the prompt must contain the word "json"), then validate
  after parsing.
- **Tool calling is a loop (07).** The model asks to call a function, you run it,
  you feed the result back, and it answers grounded in that result. Swap the fake
  `search_docs` for your vector store and you have agentic retrieval.

## Run it

```bash
# Node.js / TypeScript
npm install
node --loader ts-node/esm src/01-basic-call.ts
node --loader ts-node/esm src/06-structured-output.ts
node --loader ts-node/esm src/07-tool-calling.ts
# ...one file per example

# Go
cd go
go run . basic      # 01
go run . multiturn  # 02
go run . retry      # 03
go run . stream     # 04
go run . cost       # 05
go run . structured # 06
go run . tools      # 07
```

The Go port uses only the standard library (`net/http` + `encoding/json`) — no
SDK — so you can see exactly what goes over the wire.
