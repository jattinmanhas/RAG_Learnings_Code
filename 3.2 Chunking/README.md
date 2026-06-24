# 3.2 Chunking

After raw text is extracted (see **3.1 Document Preprocessing**), it must be split into
**chunks** before embedding. This folder implements every common chunking strategy in
TypeScript, heavily commented, with a runnable demo.

## Run it

```bash
cd nodejs
npm install
npm start
```

This runs all 8 strategies on the same sample text so you can compare their output.

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

## Core tension

- **Too small** → each chunk lacks context, meaning fragments.
- **Too large** → imprecise retrieval, wasted context window.
- **No overlap** → a fact split across a boundary is lost to both chunks.

Rule of thumb: start with **recursive** chunking, ~500 tokens per chunk, ~10–20% overlap,
and only reach for semantic/structure-aware approaches when retrieval quality demands it.

## Notes for production

- **Token** chunking here approximates `1 token ≈ 4 chars`. Swap in a real tokenizer
  (`js-tiktoken` for OpenAI, the model's HF tokenizer otherwise).
- **Semantic** chunking takes an `embed` callback so the code stays dependency-free.
  Pass your real embedding model (OpenAI / Cohere / local). It costs an embedding call
  per sentence *before* chunking — use it deliberately.
- **Sentence** splitting uses a naive regex; real systems use an NLP sentence tokenizer
  that understands abbreviations like "Dr." and decimals like "3.14".
