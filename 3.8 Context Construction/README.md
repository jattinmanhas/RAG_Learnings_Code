# 3.8 Context Construction

Retrieval hands you a **ranked list of chunks**. The model sees a **string**. Everything
between those two facts is context construction — and it is where a good retriever
quietly becomes a bad answer: evidence that didn't fit, evidence buried in the middle of
the window where attention is weakest, or evidence the model can't point back at.

This folder implements the four stages in TypeScript and Go, heavily commented, with a
runnable demo — plus the **Complete Context Construction Pipeline** that composes them
in the order that actually works.

```
ranked chunks
   │
   ├─ 1. PACKING       fit maximum relevant content under the token budget
   ├─ 2. ORDERING      strongest evidence at the top AND bottom (lost-in-the-middle)
   ├─ 3. COMPRESSION   trim / summarise so more evidence fits
   └─ 4. FORMATTING    numbered, provenance-tagged blocks → citable answers (3.9)
                │
                └─→ prompt
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

No API key needed — the tokenizer is `chars/4` and the summariser is a lexical mock. The
demo carries one realistic retrieval result (8 chunks: a near-duplicate, one 209-token
monster, four hits from the same file) through every stage under a deliberately tiny
1000-token window, then runs the full pipeline and prints the finished prompt.

## 1. Context Packing — fitting maximum relevant content within token limits

The context window is a **budget, not a container**. The system prompt, the query, the
chat history and the *reserved output* all spend from the same pool — packing to the
model's context size is the single most common way to get your answer truncated.

| Subtype | What it does | Catch |
|---------|--------------|-------|
| **Evidence budget** | `window − system − query − answer reserve − margin` | compute this *first*; everything else packs into what's left |
| **Greedy by score** | take chunks in relevance order while they fit | use `continue`, not `break` — one oversized chunk must not block three small ones |
| **Density (knapsack)** | rank by `score / tokens` | fits more distinct evidence; biases toward short chunks, so pin the top-1 by relevance |
| **Near-duplicate removal** | drop chunks with Jaccard ≥ τ | free budget; duplicates also bias the model — it reads repetition as importance |
| **Source diversity cap** | ≤ N chunks per file/document | stops one file eating the window; essential for "how does X interact with Y" |
| **Tiered packing** | top-K verbatim, tail compressed | precision where it matters, recall where it's cheap |

In the demo, greedy skipping the 209-token `pgvector` chunk buys **two extra chunks**, and
dedupe alone returns ~56 tokens for free.

## 2. Context Ordering — highest relevance at top and bottom, not in the middle

**Lost in the middle** (Liu et al., 2023): with identical evidence in the window, accuracy
peaks when the answer sits at the **start** or the **end** of the context and drops
sharply when it's buried in the middle — a U-shaped curve. Emission order is therefore a
retrieval decision, not a cosmetic one, and plain score-descending wastes the strong tail
position on your *weakest* chunk.

| Subtype | Layout | When to use |
|---------|--------|-------------|
| **Relevance descending** | 1, 2, 3, … K | the naive baseline — strong head, wasted tail |
| **Bookend / "V"** | 1, 3, 5, … 6, 4, 2 | default above ~5 chunks: ranks 1 & 2 occupy both attention peaks |
| **Document order** | group by source, restore reading order | narrative/sequential chunks (code files, tutorials) — contiguous runs read as a document |
| **Grouped bookend** | coherent per-source runs, groups at the peaks | the production compromise; best of the two above |
| **Query-adjacent** | instructions + query → context → **query restated last** | the question in the final slot beats asking once before a long context |

## 3. Context Compression — summarizing or trimming chunks before injection

Compression trades fidelity for budget. The rule that keeps it safe: **never compress the
evidence you expect to quote.** Compress the supporting cast so the star witness fits
verbatim.

| Subtype | Ratio | Quotable after? | Notes |
|---------|-------|-----------------|-------|
| **Extractive sentence selection** | 2–4× | ✅ verbatim | query-overlap scoring, reading order restored — start here |
| **Sentence-boundary truncation** | tunable | ✅ verbatim | never cut mid-clause; a severed sentence invites the model to finish it |
| **Cross-chunk sentence pruning** | 1.2–2× | ✅ verbatim | catches the repeated sentence that survives chunk-level dedupe |
| **Abstractive (LLM) summary** | 5–10× | ❌ paraphrase | highest ratio, costs a call, must be labelled so the model doesn't "quote" it |
| **Compression cascade** | as needed | mixed | run the cheap stages first, stop the moment it fits — free when evidence already fits |

The demo's cascade goes `506 → 487 → 322 → 217 → 169` tokens against a 179-token budget,
tagging the top hit as protected the whole way down.

## 4. Citation-ready formatting (the handoff to 3.9)

A model can only cite what you gave it a **handle** for. Three requirements:

1. **A stable marker per block** — `[1]`, `[2]`, … so a citation is one token to emit.
2. **Provenance next to the marker** — `path="docs/hnsw-tuning.md:L12-L28"`, so the
   answer's `[1]` resolves to a file and line range without a side lookup.
3. **Hard delimiters** — `<source>` tags fence evidence off from instructions, survive
   chunks containing markdown/code, and stop chunk text from impersonating the prompt.

Compressed chunks are labelled by *kind*: `kind="excerpt"` (shortened but verbatim — safe
to quote) vs `kind="summary"` (LLM paraphrase — cite, never quote). Without that
distinction the model will happily "quote" wording that only your summariser wrote.

```xml
<context>
<source id="1" path="docs/hnsw-tuning.md:L12-L28">
To make HNSW search faster without hurting recall, raise efSearch gradually…
</source>
<source id="2" path="docs/pgvector.md:L60-L120" kind="summary">
IVFFlat remains available and builds far faster, but its recall degrades… (key: faster, recall)
</source>
</context>
```

`resolveCitations()` maps the `[n]` markers back to chunks after generation — and flags
markers that point at nothing, which is 3.7's groundedness problem in a form you can
catch with a regex. 3.9 picks it up from there.

## 5. The Complete Context Construction Pipeline

Order of operations matters, and it is **not** the order the concepts are taught in:

```
dedupe → cap per source → compress to fit → pack under budget
       → order for attention → format with citation markers → assemble query-adjacent
```

- **Compress before packing** — compression changes what fits.
- **Pack before ordering** — ordering is meaningless on chunks that get dropped.
- **Format last** — markup is ~40% overhead on short chunks and must be inside the
  budget you measured.

The demo prints the full trace for one query:

```
budget: window=1000 − system/instructions − query×2 − answer=400 ⇒ 179 tok for evidence
dedupe: 8 → 7 chunks
source cap (≤3/source): 7 chunks, 506 tok
  compress: extractive (tail) → 322 tok
  compress: LLM summaries (tail) → 217 tok
  compress: still over — drop lowest-value chunks → 169 tok
pack: 5/5 chunks kept, 169/179 tok used
order (grouped-bookend): c1 → c2 → c6 → c5 → c4
format: 5 citable sources, 263 tok with markup
prompt: 458 tok total (answer reserve 400 still free)
```

## Rules of thumb

- Reserve the answer tokens **before** you pack. Truncated generations look like model
  failures and are almost always budget bugs.
- Use the real tokenizer. `chars/4` is fine for a demo; guessing low truncates, guessing
  high wastes budget on every single request.
- Dedupe first — it is the cheapest token you will ever buy back.
- Never put your best chunk in the middle, and never leave the last slot to your worst.
- Protect the top-1 chunk from every compressor: it's the one the answer will quote.
- Label paraphrases. An unlabelled summary is a quote the model will invent.
- Measure the *formatted* prompt, not the chunk text — markup counts.
