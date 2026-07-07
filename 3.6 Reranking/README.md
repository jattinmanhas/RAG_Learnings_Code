# 3.6 Reranking

Retrieval (see **3.5 Retrieval Optimization**) is fast but coarse. **Reranking** is the
second stage: take the top-K candidates from retrieval and re-score them with a slower,
more accurate model before building the context. This folder implements every common
reranking strategy in TypeScript and Go, heavily commented, with a runnable demo.

```
corpus (1M docs) --stage 1: bi-encoder/BM25--> top 50 --stage 2: reranker--> top 5
```

## Run it

```bash
# TypeScript
cd nodejs
npm install
npm start

# Go
cd go
go run .
```

No API key needed — the "models" are lexical mocks that demonstrate each strategy's
behaviour. The demo retrieves candidates with a bi-encoder (imperfect order, including a
keyword-trap document), then shows how each reranker reorders them.

## Why two stages?

| | Bi-encoder (stage 1) | Cross-encoder (stage 2) |
|---|---|---|
| Sees query & doc | **separately** — compares vectors | **together** — one joint forward pass |
| Doc representation | precomputed offline | computed per query |
| Cost per query | 1 embed + ANN search over millions | K model calls (K ≈ 20–100) |
| Quality | coarse — misses negation, intent, "mentions vs answers" | high — understands the pair |

A bi-encoder can't distinguish *"doc that mentions HNSW, faster, and recall"* from
*"doc that tells you how to make HNSW faster"*. A cross-encoder can — but you can only
afford to run it on a shortlist. Two-stage retrieval is the production standard because
it buys cross-encoder quality at bi-encoder scale.

## The strategies & when to use them

| # | Strategy | How it scores | Cost | Use when |
|---|----------|---------------|------|----------|
| 1 | **Cross-encoder** ⭐ | joint (query, doc) forward pass | K model calls (batchable) | **default**; best quality/latency balance |
| 2 | **Late interaction (ColBERT)** | per-token MaxSim, docs precomputed | cheap at query time, ~100× storage | large K, tight latency, storage available |
| 3 | **LLM pointwise** | LLM grades each doc 0–10 | K LLM calls (parallel) | need absolute scores for "I don't know" thresholds |
| 4 | **LLM pairwise** | LLM duels: "A or B?" | O(K²) full, O(K log K) tournament | small K, maximum precision at the top |
| 5 | **LLM listwise (RankGPT)** | one prompt ranks all K passages | 1 LLM call | best quality per dollar; watch position bias |
| 6 | **MMR** | relevance − similarity to already-picked | vector math only | corpus has near-duplicates; diversify context |
| 7 | **RRF** | 1/(k + rank) summed across lists | free | fusing multiple retrievers; baseline to beat |
| 8 | **Business rules** | relevance × recency × source weight | free | freshness/authority matters (docs vs old forum) |

## Production recipe

```
hybrid retrieval (top 50)
  → cross-encoder rerank (top 10)
  → MMR / business rules
  → top 3–5 into the context window
```

1. **Retrieve wide** — stage 1's job is recall. If the right doc isn't in the top-50,
   no reranker can save you.
2. **Rerank narrow** — stage 2's job is precision at the top.
3. **Then shape** — MMR for diversity and business rules for freshness/authority run
   last, on already-relevant results.

## Real rerankers to swap in

The mocks in `index.ts` / `main.go` implement the same function signatures as real models:

- **Hosted cross-encoder APIs** — Cohere Rerank 3.5, Voyage `rerank-2`, Jina Reranker.
  One HTTPS call scores all candidates in a batch; simplest production path.
- **Open-weight cross-encoders** — `BAAI/bge-reranker-v2-m3`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (via sentence-transformers or ONNX). Self-host
  when data can't leave your infra.
- **Late interaction** — ColBERTv2 / the RAGatouille library; pgvector can store the
  token vectors, or use Vespa/Qdrant multi-vector support.
- **LLM judges** — any cheap, fast model (Claude Haiku, GPT mini-class) with a
  temperature-0 rubric prompt for pointwise; RankGPT-style prompts for listwise.

## Gotchas

- **Reranking cannot add recall.** It only reorders what stage 1 found. Fix recall with
  hybrid retrieval and query rewriting (3.3, 3.5), not with a better reranker.
- **Don't rerank the whole corpus.** Cross-encoder cost is linear in candidates; keep
  K in the 20–100 range.
- **Listwise LLMs have position bias** — they favor passages that appear early in the
  prompt. Shuffle input order, or average over two passes.
- **Normalize before thresholding.** Cross-encoder scores are model-specific and not
  comparable across models — recalibrate any "confidence" thresholds when you swap models.
- **Latency budget:** hosted rerank APIs add ~50–200 ms. If that's too much, late
  interaction or a distilled MiniLM cross-encoder on-box is the usual escape hatch.
