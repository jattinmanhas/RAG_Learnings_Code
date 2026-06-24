# 3.3 Query Rewriting

After chunking and indexing (see **3.2 Chunking**), the bottleneck in most RAG
pipelines is **retrieval quality**, not generation quality.  Users write short,
casual questions; your vector index holds long, formal passages.  Query rewriting
transforms the raw question into one or more forms that retrieve **better
documents** before any LLM sees them.

## Run it

```bash
# Node.js / TypeScript
cd nodejs
npm install
npm start

# Go
cd go
go run .
```

Both implementations use a **mock LLM** — no API key or network required.
Every strategy prints its rewritten query/queries to stdout so you can compare
them side-by-side.

## The strategies & when to use them

| # | Strategy | LLM? | What it produces | Use when |
|---|----------|-------|-----------------|----------|
| 1 | **Keyword Expansion** | ✗ | synonyms injected inline | narrow domain with stable vocabulary; zero latency needed |
| 2 | **HyDE** | ✓ | hypothetical answer passage to embed | question style ≠ document style; dense retrieval |
| 3 | **Multi-Query** | ✓ | N paraphrase variants → union of results | underspecified or multi-interpretation queries |
| 4 | **Step-Back** | ✓ | abstract/background question alongside original | questions that need foundational context |
| 5 | **Sub-query Decomposition** | ✓ | atomic sub-questions → per-question retrieval | complex, comparative, or multi-part questions |
| 6 | **RAG-Fusion** | ✓ | multi-query + Reciprocal Rank Fusion | best overall precision/recall; multiple retrievers |
| 7 | **Contextual Compression** | ✓ | standalone query from chat history | **always** in multi-turn / conversational RAG |

## How each strategy works

### 1. Keyword Expansion (no LLM)
Inject synonyms for known terms using a hand-crafted map.  Fast, deterministic,
zero API cost.  Best as a cheap pre-filter or in narrow domains (legal, medical).

### 2. HyDE — Hypothetical Document Embeddings
Ask the LLM: *"Write a short passage that answers this question."*  Embed the
**fake answer** instead of the bare question.  The hypothetical lives in the same
vector space as real documents, so nearest-neighbour search retrieves documents
similar to "what the answer looks like."

> Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels*
> (arxiv 2212.10496)

### 3. Multi-Query
Generate N paraphrases of the query, run retrieval for each, take the **union**.
Increases recall — different wordings hit different parts of the index.  Combine
with RAG-Fusion to avoid diluting precision.

### 4. Step-Back Prompting
Rephrase the specific question into a broader, more abstract question that
retrieves **background knowledge**.  Use *both* queries: the specific one for
facts, the step-back one for context.

> Zheng et al., 2023 — *Take a Step Back: Evoking Reasoning in LLMs via
> Abstraction* (arxiv 2310.06117)

### 5. Sub-query Decomposition
Break a complex question into atomic sub-questions.  Retrieve separately for each,
then hand all evidence to the generator.  Essential for comparative or multi-part
questions ("Compare X and Y across three dimensions").

### 6. RAG-Fusion
Multi-Query + **Reciprocal Rank Fusion (RRF)**.

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
```

RRF re-ranks the union of results so that documents appearing highly across
*many* query variants win.  `k = 60` is the standard constant (empirically best
across most benchmarks).  Also the standard way to fuse BM25 + dense results.

> Cormack et al., SIGIR 2009 — *Reciprocal Rank Fusion outperforms Condorcet*

### 7. Contextual Compression
In multi-turn chat, the user says *"What about its scalability?"* — "its"
references PostgreSQL from three turns ago.  The retriever has no memory.
Contextual compression rewrites the latest message into a **standalone question**
before retrieval, then passes the full history to the generator.

Apply this **always** in conversational RAG — one LLM call, large precision gain.

## Plug in a real LLM

All LLM-dependent strategies accept an `LLMFn` / `LLMFn` callback:

```typescript
// TypeScript
import Anthropic from "@anthropic-ai/sdk";
const client = new Anthropic();
const realLLM: LLMFn = async (prompt) => {
  const msg = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 512,
    messages: [{ role: "user", content: prompt }],
  });
  return (msg.content[0] as { text: string }).text;
};
```

```go
// Go
import anthropic "github.com/anthropics/anthropic-sdk-go"
client := anthropic.NewClient()
realLLM := func(ctx context.Context, prompt string) (string, error) {
    msg, err := client.Messages.New(ctx, anthropic.MessageNewParams{
        Model:     anthropic.ModelClaudeHaiku4_5,
        MaxTokens: 512,
        Messages:  []anthropic.MessageParam{{Role: "user", Content: prompt}},
    })
    return msg.Content[0].Text, err
}
```

## Recommended defaults

| Scenario | Strategy |
|----------|----------|
| Single-turn, simple query | **HyDE** or **Multi-Query (n=3)** |
| Complex / comparative question | **Sub-query Decomposition** |
| Multi-turn chat | **Contextual Compression** (always) |
| Multiple retrievers / need best precision | **RAG-Fusion** |
| Domain with fixed vocabulary | **Keyword Expansion** as pre-filter |
| Need background knowledge | **Step-Back** alongside original |

Combine freely — e.g. contextual compression → multi-query → RAG-Fusion is a
common production pattern.
