# 3.2 Chunking

After raw text is extracted (see **3.1 Document Preprocessing**), it must be split into
**chunks** before embedding. This folder implements every common chunking strategy in
TypeScript **and Go**, heavily commented, with runnable demos.

## Run it

```bash
cd nodejs
npm install
npm start
```

```bash
cd go
go run .
```

This runs all 9 strategies on the same sample text (the AST strategy runs on a code
sample) so you can compare their output.

## The strategies & when to use them

| # | Strategy | Cuts on… | Use when |
|---|----------|----------|----------|
| 1 | **Fixed-size** | every N chars | quick baseline; unstructured text (logs, OCR) |
| 2 | **Fixed + overlap** | every N chars, repeating a tail | default for fixed approaches; stops boundary loss |
| 3 | **Sentence** | sentence boundaries | well-punctuated prose; want whole thoughts |
| 4 | **Paragraph** | blank lines | docs structured with blank lines; respect author intent |
| 5 | **Recursive** ⭐ | biggest natural boundary that fits | **best general default** for mixed/unknown content |
| 6 | **Token** | token budget | must respect a hard token limit / control cost |
| 7 | **Markdown-header** | `#` headings | docs/wikis with headings; want section metadata |
| 8 | **Semantic** | where meaning shifts | high-value corpora; topic-drifting text; costlier |
| 9 | **AST / code-aware** | function/class/method boundaries | **source code — the only correct choice for codebase RAG** |

## Core tension

- **Too small** → each chunk lacks context, meaning fragments.
- **Too large** → imprecise retrieval, wasted context window.
- **No overlap** → a fact split across a boundary is lost to both chunks.

Rule of thumb: start with **recursive** chunking, ~500 tokens per chunk, ~10–20% overlap,
and only reach for semantic/structure-aware approaches when retrieval quality demands it.
**Exception: source code.** Prose chunkers cut functions in half — for code, go straight
to AST chunking.

## AST chunking for codebase RAG

Prose strategies slice through the middle of functions and glue unrelated declarations
together, and half a function embeds as noise. AST chunking parses the source and cuts on
**syntactic boundaries** instead, so every chunk is a complete unit of code:

1. Parse the file into an AST; walk top-level declarations.
2. One chunk per function/class/type (doc comments ride along with their declaration).
3. A class too big for the budget splits **by method**, with the class header prepended
   so the method keeps its context.
4. Every chunk carries metadata — `filePath`, `symbol`, `kind`, `startLine`/`endLine` —
   which later powers citations ("see `users.ts:42`") and filtered retrieval.

The TypeScript version uses the **TypeScript compiler API** (already a dev dependency);
the Go version uses stdlib **`go/parser`**. Both are real production-grade ASTs but only
parse their own language — for a multi-language codebase use **tree-sitter**
(`web-tree-sitter` in Node, `github.com/smacker/go-tree-sitter` in Go): the chunking
logic stays identical, only the parser swaps.

Also worth knowing for codebase RAG: exact identifier lookups (`getUserById`) are a
*keyword* problem, not an embedding problem — pair AST chunks with the hybrid BM25 +
vector retrieval from **3.5 Retrieval Optimization**.

## Notes for production

- **Token** chunking here approximates `1 token ≈ 4 chars`. Swap in a real tokenizer
  (`js-tiktoken` for OpenAI, the model's HF tokenizer otherwise).
- **Semantic** chunking takes an `embed` callback so the code stays dependency-free.
  Pass your real embedding model (OpenAI / Cohere / local). It costs an embedding call
  per sentence *before* chunking — use it deliberately.
- **Sentence** splitting uses a naive regex; real systems use an NLP sentence tokenizer
  that understands abbreviations like "Dr." and decimals like "3.14".
