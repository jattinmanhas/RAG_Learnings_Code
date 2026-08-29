# 3.9 Citations

3.8 handed the model numbered, provenance-tagged sources and told it how to
cite them. What comes back is a string with `[n]` markers sprinkled through
it — and a marker is a *claim*, not a fact. It can point at nothing, point at
a real source that has nothing to do with the sentence, or point at a real
source that says the opposite of what the sentence claims. This folder is the
pipeline that turns that raw, possibly-wrong answer into one you can show a
user with confidence, in TypeScript and Go, heavily commented, with a
runnable demo that hits every failure mode in one pass.

```
raw answer with [n] markers
   │
   ├─ 1. PARSING        split into sentence + marker pairs
   ├─ 2. VERIFICATION   does [n] exist, AND does it SUPPORT this sentence?
   ├─ 3. REPAIR         strip / flag / drop citations that fail either check
   ├─ 4. ATTRIBUTION    best-guess citation for sentences with none at all
   └─ 5. RENDERING      inline markers + a deduplicated reference list
                │
                └─→ answer you can show a user
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

No API key needed — the support checker is lexical word-overlap, a stand-in
for an NLI model or an LLM judge. The demo carries one model answer through
every stage: a citation that's real and supports its sentence, one that's
real but off-topic, one that's real but *contradicts* its source, one that
points at nothing, one sentence the model forgot to cite that a source
clearly backs, and one sentence nothing backs at all.

## 1. Citation Parsing — sentence + marker pairs

A model cites in clusters (`"...faster [1][3]."`) and inline, so parsing has
to split on **sentence** boundaries, not marker boundaries — one sentence can
carry several citations — then strip the markers back out of the display
text so they don't get rendered twice.

```
cites=[1]  "Raise efSearch gradually and re-measure recall on a held-out set."
cites=[-]  "Recall should always be measured against exact brute-force search on the same query set."
```

## 2. Citation Verification — existence AND support

3.8's `resolveCitations()` only checked **existence**: does `n` appear in the
map? That catches a hallucinated marker number, but not the more common and
more dangerous failure — a model citing a **real** source that has nothing to
do with the sentence it's attached to. Verification checks both.

| Result | Meaning | Example |
|---|---|---|
| **verified** | real source, clears the support threshold | `[1]` cites the exact efSearch-tuning passage |
| **unsupported** | real source, but off-topic for this claim | `[4]` (HNSW's graph structure) cited for a claim about time-series forecasting |
| **missing** | no such source — a hallucinated marker | `[7]` doesn't exist in the source map at all |

The support checker is injected (`SupportChecker` / `SupportChecker`), so the
lexical mock here is swappable for a real NLI model or an LLM judge without
touching anything else.

**The blind spot, on purpose:** lexical overlap flags an *off-topic* citation
reliably, but not a *contradicting* one — "recall degrades" and "recall never
degrades" share every content word. The demo's `[3]` sentence is built to
trip exactly this: it reuses the source's own wording almost verbatim while
negating it, and the lexical checker calls it **verified** anyway. Production
systems pair the cheap lexical pass with [3.7](../3.7%20Retrieval%20Confidence)'s
LLM groundedness check before trusting a citation on anything that matters.

## 3. Citation Repair — what to do with a citation that fails

A missing marker always gets discarded — it points at nothing, so there's no
judgment call. An unsupported one is genuinely ambiguous: the sentence might
still be true, just mis-cited.

| Strategy | Behaviour | Use when |
|---|---|---|
| **strip** | drop the bad marker(s), keep the sentence as an uncited claim | low stakes, completeness over provenance |
| **flag** | keep the sentence, append a visible caveat | the default — surface the problem, don't hide the claim |
| **drop** | remove the whole sentence | high-stakes answers where an unverifiable claim is worse than a gap |

## 4. Retroactive Attribution — citations for sentences that have none

Not every model obediently cites everything, and repair (3) can strip a
sentence down to zero citations. Rather than ship an uncited claim,
attribution finds the source the sentence most resembles and attaches it —
but as `inferred`, never merged into `cites`, because the model never claimed
it. In the demo, the brute-force-recall sentence backfills cleanly to a real
source; the "this project uses TypeScript and Go" sentence stays uncited,
because nothing should be invented when nothing actually matches.

## 5. Citation Rendering — inline markers + a reference list

The reference list only lists sources actually used — not the whole
retrieval set 3.8 packed — and an inferred citation is marked with a trailing
`~` so a reader (or a downstream check) can tell "the model said this" from
"the system's best guess" apart at a glance.

```
Raise efSearch gradually and re-measure recall on a held-out set. [1]
...
Recall should always be measured against exact brute-force search on the same query set. [5]~

References:
[1] docs/hnsw-tuning.md:L12-L28
[3] docs/pgvector.md:L60-L120 (summary — paraphrase, do not quote)
[5] docs/hnsw-tuning.md:L46-L58
```

## 6. The Complete Citation Pipeline

`parse → verify → repair → backfill → render`, tracked with a **coverage**
score: the fraction of the final answer's sentences carrying at least one
citation, model-claimed or system-inferred. Coverage is a cheap, immediate
proxy for how much of an answer you can actually stand behind — track it
alongside 3.7's groundedness signals, not instead of them.

```
parse: 7 sentences
verify: 1 hallucinated marker(s), 1 unsupported citation(s)
repair (flag): 0 sentence(s) dropped, 2 flagged
backfill: 1 previously-uncited sentence(s) attributed
coverage: 4/7 sentences carry a citation (57%)
```

## Rules of thumb

- Existence is not support. A marker resolving to a real source only means
  the model didn't invent a number — it says nothing about whether that
  source backs the sentence it's attached to.
- Lexical overlap catches "wrong source," not "wrong conclusion." A
  contradiction that reuses the source's own words sails through — escalate
  to an entailment model or an LLM judge before trusting high-stakes claims.
- Never merge inferred citations into the model's claimed ones. A reader
  needs to know which is which, and so does anything auditing the answer
  later.
- Attribution fills gaps, it doesn't override — a sentence the model already
  cited is never re-attributed, even if a better-matching source exists.
- `drop` beats `flag` beats `strip` as stakes rise: silently keeping an
  unverifiable claim's marker is the worst of the three, because it *looks*
  cited.
- Measure coverage on the answer you actually render, after repair and
  backfill — not on the model's raw output, which overstates it.
