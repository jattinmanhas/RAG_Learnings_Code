# 3.7 Retrieval Confidence

Retrieval **always returns something** — top-K nearest neighbours exist even when the
corpus contains nothing relevant. Retrieval confidence is the discipline of asking
*"should I trust what came back?"* before (and after) generating an answer. It is what
lets a RAG system say **"I don't know"** instead of hallucinating from garbage context.

This folder implements the three families of confidence signals in TypeScript and Go,
heavily commented, with a runnable demo — plus a **complete confidence pipeline** that
composes them cheapest-first.

```
query → retrieve top-K
          │
          ├─ 1. absolute thresholds        (free, crude, corpus-specific)
          ├─ 2. relative/distributional     (free, more robust)
          └─ 3. LLM groundedness checks     (most robust, costs LLM calls)
                    │
                    └─→ ANSWER / HEDGE / REFUSE
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

No API key needed — embeddings and LLM judges are lexical mocks. The demo runs three
queries against a small vector-DB knowledge base: one **answerable**, one **ambiguous**
(partial coverage), one **unanswerable** (off-corpus), and shows what each signal says.

## The three families & their subtypes

### 1. Absolute thresholding (crude, corpus-specific)

Compare a raw score against a fixed cutoff. Free and simple, but the "right" number
depends on the embedding model, the metric, and the corpus — it does not transfer.

| Subtype | Rule | Catch |
|---------|------|-------|
| **Similarity floor** | drop hits with cosine < τ (e.g. 0.30) | τ ≈ 0.3 for one model is ≈ 0.7 for another |
| **Top-1 floor** | if even the best hit < τ, refuse outright | cheapest possible "I don't know" gate |
| **Count-above-threshold** | need ≥ N hits above τ to attempt an answer | guards against a single lucky match |
| **Calibrated threshold** | pick τ from a labelled dev set (e.g. 95th percentile of irrelevant-pair scores) | best absolute variant; needs labels & re-calibration per model/corpus |
| **Reranker-score floor** | threshold the cross-encoder / rerank score instead of cosine | rerank scores are better calibrated than raw cosine (see 3.6 pointwise LLM) |

### 2. Relative / distributional signals (more robust)

Ignore the absolute numbers; look at the **shape** of the score distribution across the
top-K. Shape transfers across models and corpora much better than raw values.

| Subtype | Signal | Reads as |
|---------|--------|----------|
| **Top-1/Top-2 gap** | `score₁ − score₂` | big gap ⇒ one clear winner; flat ⇒ nothing stands out |
| **Relative drop-off** | `score₁ / mean(score₂..K)` | ratio ≫ 1 ⇒ confident; ≈ 1 ⇒ all hits equally mediocre |
| **Softmax entropy** | entropy of softmax(scores) | low entropy ⇒ peaked (confident); high ⇒ uniform (lost) |
| **Z-score vs corpus** | `(score₁ − μ_random) / σ_random` | is the top hit an outlier vs typical background similarity? |
| **Cross-retriever agreement** | overlap of dense & BM25 top-K (Jaccard / RRF mass) | two independent retrievers agreeing is strong evidence |

### 3. LLM-based groundedness checks (most robust, most expensive)

Ask a model to *read* the evidence. Catches the failure vector math can't see: a chunk
can be topically similar yet **not actually answer** the question.

| Subtype | Question the LLM answers | When it runs |
|---------|--------------------------|--------------|
| **Answerability / relevance judge** | "Can these passages answer this query? YES / PARTIAL / NO" | pre-generation gate |
| **Per-passage grading** | "Grade each passage 0–10 for whether it answers the query" | pre-generation filter (drop < cutoff) |
| **Claim-level groundedness (NLI-style)** | "Is each claim in the draft answer supported by the cited passage?" | post-generation verification |
| **Citation verification** | "Does cited chunk X actually contain the quoted fact?" | post-generation |
| **Self-consistency** | sample the answer N times; do they agree? | post-generation, N× cost |

## 4. The Complete Confidence Pipeline

Compose the signals **cheapest-first**, exiting early whenever a cheap signal is
decisive. This buys LLM-grade robustness at near-vector-math average cost:

```
retrieve top-K (+ BM25 list for agreement)
  │
  ├─ Stage 1: absolute floor          top-1 < τ_hard        → REFUSE (fast exit)
  ├─ Stage 2: distributional check    gap/entropy/agreement → confident? skip to generate
  │                                                          hopeless? REFUSE
  ├─ Stage 3: LLM answerability       (only for the gray zone) NO → REFUSE, PARTIAL → HEDGE
  ├─ Stage 4: generate with citations
  └─ Stage 5: groundedness check      unsupported claims → strip / hedge / regenerate
  │
  └─→ ANSWER (confident) | HEDGE ("based on limited info…") | REFUSE ("I don't know")
```

The demo (`pipeline` section of the code) runs all three queries through this cascade
and prints which stage decided, and why.

## Rules of thumb

- Never ship raw-cosine thresholds without a dev set — they silently break when you
  change embedding models.
- Distributional signals are the best free upgrade; start with the top-1/top-2 gap.
- Reserve LLM checks for the **gray zone** — most queries should exit at stage 1–2.
- "I don't know" is a feature. Measure your refusal precision/recall like any classifier.
