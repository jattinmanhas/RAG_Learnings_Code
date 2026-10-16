# Production GenAI Engineer Roadmap
### Backend Engineer → GenAI Engineer / AI Backend Engineer

> **Goal:** Build production-grade AI systems — not toy demos.

---

## Table of Contents

- [Phase 0 — Prerequisites](#phase-0--prerequisites)
- [Phase 1 — LLM Application Fundamentals](#phase-1--llm-application-fundamentals)
- [Phase 2 — Embeddings & Vector Search](#phase-2--embeddings--vector-search)
- [Phase 3 — RAG Engineering](#phase-3--rag-engineering)
- [Phase 4 — AI Behavior Engineering](#phase-4--ai-behavior-engineering)
- [Phase 5 — Agents & Tool Use](#phase-5--agents--tool-use)
- [Phase 6 — AI Security & Safety](#phase-6--ai-security--safety)
- [Phase 7 — Production AI Engineering](#phase-7--production-ai-engineering)
- [Phase 8 — Frameworks](#phase-8--frameworks)
- [Phase 9 — Advanced Model Customization](#phase-9--advanced-model-customization)
- [Capstone Project](#capstone-project)
- [Highest ROI Topics](#highest-roi-topics)

---

# Phase 0 — Prerequisites

## Already Strong

- REST APIs
- Node.js
- Go
- PostgreSQL
- Redis
- Docker
- Authentication
- Microservices
- SSE

## Keep Practicing

- DSA
- System Design
- Backend Architecture

---

# Phase 1 — LLM Application Fundamentals

**Objective:** Understand how modern LLM applications actually work.

---

## 1.1 Tokens & Context Windows

### Learn
- Tokenization (how BPE works)
- Context windows
- Context truncation strategies
- Token pricing model
- Conversation history management

### Understand
- Why context matters — context is the model's only memory
- Why long conversations become expensive
- The lost-in-the-middle problem — model attention degrades for content in the middle of a long context
- Why RAG exists as a solution to context limits

---

## 1.2 Model Parameters

### Learn
- Temperature
- Top-P
- Max Tokens
- Stop Sequences
- Frequency Penalty
- Presence Penalty

### Understand
- Deterministic vs creative outputs
- Hallucination tradeoffs with temperature
- How stop sequences control generation boundaries

---

## 1.3 Prompt Fundamentals

### Prompt Types
- System Prompts
- Developer Prompts
- User Prompts

### Prompting Patterns
- Zero-Shot
- Few-Shot
- Chain-of-Thought (CoT)
- Structured Prompting (XML tags, delimiters)

### Instruction Design
- Constraint design
- Output formatting directives
- Behavioral guidance
- Positive vs negative instructions

---

## 1.4 Structured Outputs

### Learn
- JSON Mode
- Schema Validation
- Typed Responses
- Output Enforcement

### Understand
- Why structured outputs reduce downstream parsing failures
- When to enforce schema vs when to parse free text

---

## 1.5 Streaming Responses

### Learn
- Token Streaming
- Server-Sent Events (SSE)
- Progressive UI rendering

---

## 1.6 APIs

### Learn
- OpenAI API
- Anthropic API

### Understand
- Retry handling with backoff
- Rate limits
- Timeouts
- Cost tracking per request

---

# Phase 2 — Embeddings & Vector Search

**Objective:** Understand semantic retrieval from first principles.

---

## 2.1 Embeddings

### Learn
- What embeddings are — high-dimensional vectors representing semantic meaning
- How similarity is encoded in vector space
- Embedding generation pipeline
- Embedding dimensionality and its tradeoffs (higher = more expressive, more storage)
- Normalization — why embeddings must be normalized for cosine similarity

### Embedding Model Selection
- MTEB Benchmark — the standard leaderboard for choosing embedding models
- Domain-specific models vs general-purpose models
- Multilingual embedding models
- Size vs quality tradeoffs

---

## 2.2 Vector Search

### Learn
- Cosine Similarity
- Dot Product Similarity
- Nearest Neighbor Search (exact vs approximate)
- ANN algorithms — HNSW, IVF

---

## 2.3 Sparse vs Dense Retrieval

### Dense Retrieval
- Embedding-based
- Captures semantic meaning
- Misses exact keyword matches

### Sparse Retrieval (BM25 / SPLADE)
- Keyword/term frequency based
- Fast and interpretable
- Foundation of traditional search (Elasticsearch, Solr)
- Critical to understand before hybrid search

### Why Both Matter
- Dense alone misses exact terms (product codes, names, acronyms)
- Sparse alone misses semantic meaning
- Hybrid search combines both — you cannot design it well without understanding each

---

## 2.4 PostgreSQL + pgvector

### Learn
- Vector columns
- Similarity queries
- Indexing — IVFFlat vs HNSW
- Production schema design

---

## 2.5 Search Architectures

### Keyword Search
### Semantic Search
### Hybrid Search

Understand strengths, weaknesses, and when to combine them.

---

## 2.6 Optional Vector Databases

Know when and why you'd reach for these over pgvector:

- Pinecone
- Qdrant
- Weaviate
- Chroma

---

# Phase 3 — RAG Engineering

**Objective:** Build enterprise-grade retrieval systems.

---

## RAG Pipeline Overview

```
Ingest Documents
→ Pre-process
→ Chunk
→ Embed
→ Store

User Query
→ Transform Query
→ Retrieve
→ Re-rank
→ Construct Context
→ Generate
→ Cite Sources
```

---

## 3.1 Document Pre-processing

> This step is skipped in tutorials. It is unavoidable in production.

### Learn
- PDF parsing — text extraction, handling scanned PDFs
- OCR — for image-based documents
- Table extraction from PDFs and HTML
- Handling corrupt or malformed documents
- Metadata extraction — author, date, section, page number
- Format normalization — converting HTML, DOCX, Markdown to clean text

### Understand
- Garbage in, garbage out — retrieval quality is bounded by ingestion quality
- Why pre-processing is a standalone pipeline, not a one-liner

---

## 3.2 Chunking

### Learn
- Fixed Chunking
- Recursive Chunking
- Semantic Chunking
- Metadata-Aware Chunking
- Chunk Overlap
- AST / Code-Aware Chunking (for codebase RAG — cut on function/class boundaries, not characters)

### Understand
- How chunk size affects retrieval precision vs recall
- Why chunk boundaries matter for answer completeness
- Why prose chunkers fail on source code (half a function embeds as noise)

---

## 3.3 Query Transformation

> Most RAG failures happen here — before the model even generates.

### Query Rewriting
Rewrite the user's raw query into a cleaner, more retrieval-friendly form before embedding.

### Multi-Query Retrieval
Generate N variants of the query. Retrieve against all. Union and deduplicate results.

### HyDE (Hypothetical Document Embeddings)
1. Feed the query to the model
2. Generate a hypothetical answer (even if hallucinated)
3. Embed the hypothetical answer
4. Retrieve against that embedding

Rationale: A hypothetical answer is semantically closer to real answer-containing chunks than the raw question is.

---

## 3.4 Advanced Retrieval

### Parent-Child Retrieval
- Embed small, precise child chunks
- When a child chunk matches, retrieve its larger parent chunk
- Gives the model more context without polluting the embedding space

### Small-to-Big Retrieval
- Retrieve small chunks for precision
- Expand to surrounding context for generation

---

## 3.5 Retrieval Optimization

- **Top-K Retrieval** — dense cosine search, tuning K for the recall/precision tradeoff
- **Metadata Filtering** — narrow candidates by structured fields (source, language, access) before scoring; correctness and security, not just relevance
- **Hybrid Retrieval** — fuse dense vectors (semantic recall) with BM25 keyword search (exact-token precision) via Reciprocal Rank Fusion or weighted score blending
- **Retrieval Tuning** — the knobs: topK, dense/sparse mix (alpha), RRF constant, candidate width, score threshold
- Runnable Go + Node.js examples in `3.5 Retrieval Optimization/`

---

## 3.6 Re-Ranking

### Stage 1 — Bi-Encoder Retrieval
- Fast
- Used to retrieve top-K candidates from the full corpus
- Same model that generated the embeddings

### Stage 2 — Cross-Encoder Re-Ranking
- Slow but significantly more accurate
- Takes (query, chunk) as a pair and scores them together
- Applied only to the top-K candidates, not the full corpus

### Types of Re-Ranking
- **Cross-encoder** — joint (query, doc) scoring; the production standard
- **Late interaction (ColBERT-style)** — token-level MaxSim; middle ground between bi- and cross-encoders
- **LLM-based** — pointwise / pairwise / listwise (RankGPT-style) judging with a generative model
- **MMR** — diversity re-ranking to reduce redundancy in the final context
- **RRF** — rank fusion across multiple retrievers
- **Business-rule** — recency decay, source-authority boosts on top of relevance

### Understand
- Why two-stage retrieval is the production standard
- The latency/accuracy tradeoff between stages
- Runnable Go + Node.js examples in `3.6 Reranking/`

---

## 3.7 Retrieval Confidence & "I Don't Know" Handling

### Learn
- **Absolute thresholding** (crude, corpus-specific) — similarity floor, top-1 floor, count-above-threshold, calibrated τ from a dev set, reranker-score floor
- **Relative/distributional signals** (more robust) — top-1/top-2 gap, relative drop-off, softmax entropy, z-score vs corpus background, cross-retriever agreement
- **LLM-based groundedness checks** (most robust, most expensive) — answerability gate, per-passage grading, claim-level NLI-style groundedness, citation verification, self-consistency
- **The Complete Confidence Pipeline** — cheapest-first cascade with early exits → ANSWER / HEDGE / REFUSE

### Understand
- Hallucination often happens when retrieval fails silently — top-K always returns *something*
- Absolute cutoffs don't transfer across embedding models; distribution shape does
- Reserve LLM checks for the gray zone; most queries should decide for free at stage 1–2
- Explicit "no result" paths reduce hallucination rate significantly
- Runnable Go + Node.js examples in `3.7 Retrieval Confidence/`

---

## 3.8 Context Construction

### Learn
- **Context Packing** — fitting maximum relevant content within token limits: evidence budget (window − system − query − answer reserve), greedy vs density/knapsack packing, near-duplicate removal, per-source diversity cap, tiered packing
- **Context Ordering** — highest relevance at top and bottom, not in the middle (lost-in-the-middle): relevance descending, bookend/"V", document-order restore, grouped bookend, query-adjacent layout
- **Context Compression** — summarizing or trimming chunks before injection: extractive sentence selection, sentence-boundary truncation, cross-chunk sentence pruning, abstractive LLM summaries, the compression cascade
- **Citation-ready formatting** — numbered `<source>` blocks with path/line provenance, excerpt vs summary labelling, marker resolution (the handoff to 3.9)
- **The Complete Context Construction Pipeline** — dedupe → source cap → compress → pack → order → format → assemble

### Understand
- The context window is a budget, not a container — reserve the answer tokens *before* packing
- Emission order is a retrieval decision: score-descending wastes the strong tail slot on your weakest chunk
- Compress before packing, pack before ordering, format last (markup counts against the budget)
- Never compress the chunk you expect to quote; label paraphrases or the model will invent quotes
- Runnable Go + Node.js examples in `3.8 Context Construction/`

---

## 3.9 Citations

### Learn
- Source tracking through the pipeline
- Chunk attribution in the response
- Grounded responses with references

---

## 3.10 RAG Evaluation ⬅ YOU ARE HERE (expanded)

> The most important section in the roadmap. Every technique in 3.3–3.8 is a hypothesis until this exists.

### 3.10.1 Build the Golden Dataset
- 50–100 hand-written questions over your real corpus
- For each: expected source chunks/documents, a reference answer, and a type tag (factual, multi-hop, unanswerable, keyword-heavy)
- Include unanswerable questions deliberately — they test 3.7
- Version the dataset in git like code

### 3.10.2 Retrieval Metrics (measure first, separately)
- Recall@k — did the right chunk make it into the top-k?
- Precision@k — how much of the top-k is noise?
- MRR / nDCG — how high did the right chunk rank?
- Measure retrieval *before* generation: a bad answer from bad retrieval is a retrieval bug

### 3.10.3 Generation Metrics
- Faithfulness / groundedness — is every claim supported by the provided context?
- Answer relevance — does it answer the actual question?
- Citation accuracy — do cited sources actually contain the claim?
- Refusal correctness — does it say "I don't know" on unanswerable questions, and only those?

### 3.10.4 LLM-as-Judge
- Write explicit grading rubrics, not "rate 1–10"
- Prefer binary or small-scale judgments (supported / not supported)
- Validate the judge: hand-label ~30 examples, measure agreement with the judge
- Use a different (or stronger) model as judge than the one generating

### 3.10.5 Ablations — Go Back and Measure Phase 3
- Baseline: plain dense retrieval, no tricks
- Add one technique at a time: hybrid, reranking, query rewriting, HyDE, parent-child, compression
- Record recall, faithfulness, latency, and cost for each
- Keep only what moves the numbers on *your* data

### 3.10.6 Eval as Infrastructure
- One command runs the full suite and prints a comparison table
- Run in CI on every prompt, chunking, or retrieval change
- Store results per run so regressions are visible over time
- Tools (optional, after building it yourself once): RAGAS, DeepEval

### Understand
- Evals are the differentiator — most "GenAI" resumes have demos, few have numbers
- Top-K always returns something; only evals reveal silent retrieval failure
- A technique that helps on a benchmark can hurt on your corpus

---

# Phase 4 — Agents, Tools & Harness Engineering

**Objective:** Build systems that act, reliably — and know when *not* to use an agent.

> Merges old Phase 5, memory (old 4.4), reasoning loops (old 4.5), and agent-facing security (old 6.1, 6.3, 6.4).

---

## 4.1 Tool Calling Fundamentals
- How tool calls work: model requests, harness executes, result returns as a message
- Tool schemas: names, descriptions, input schemas
- Checking model support for tools (capabilities, chat templates, quantized builds)
- Tool choice modes: auto, required, specific tool
- Parallel tool calls and partial failures

## 4.2 Tool Design
- Descriptions say what a tool is for *and* what it's not for
- Distinct names; avoid overlapping tools
- Routing between competing tools (internal KB vs web search) via descriptions + system prompt
- Compact outputs: return only needed fields, truncate, include total counts
- Aggregation tools vs raw data (do the math in the database, not the model)
- Bounded fallback tools for the long tail (whitelisted fields, row caps)
- Validate every argument as untrusted input

## 4.3 The Harness & Agent Loop
- Loop structure: call model → check for tool calls → execute → append result → repeat
- Stop conditions: no tool call, final-answer tool, verification passed
- Safety stops: max steps, repeated identical calls, error budget, token budget
- Error handling: return errors as tool results so the model can recover
- Verification inside the loop: run tests / validate schema, feed failures back
- Build one by hand (~50–100 lines) before touching any framework

## 4.4 Workflows vs Agents
- Agentic workflows: stages fixed in code, model decides inside chosen stages
- Autonomous agents: model chooses steps and tools
- Common patterns: routing, prompt chaining, parallelization, orchestrator-workers, evaluator-optimizer
- Default to the simplest workflow that passes evals; add autonomy only where needed

## 4.5 Loop Engineering
- Turn-based loops (user drives each turn)
- Goal-based loops (explicit, verifiable stop condition + turn cap)
- Time-based loops (interval or schedule)
- Proactive loops (event-triggered, no human in real time)
- Choosing trigger, stop condition, verification, and cost limits per task

## 4.6 Context Engineering for Agents
- Context grows every step; tool results are re-sent on every call
- Trimming and summarizing old tool results
- Rolling summary + recent verbatim turns
- Token budget: system prompt and current task are never dropped

## 4.7 Memory Systems
- Short-term (session history) vs long-term (persistent) vs episodic (past actions and outcomes)
- Store → retrieve → inject → update
- Always-on core memory vs threshold-based retrieval vs memory-as-a-tool
- Staleness, corrections, deletion, privacy

## 4.8 Agentic RAG
- Retrieval as a tool instead of a fixed pipeline step
- Query retry / rewrite when retrieval is weak
- Multi-source retrieval (internal KB + web + database)
- Compare against the fixed pipeline from Phase 3 using 3.10 evals

## 4.9 Skills & Hooks
- Skills: `SKILL.md` procedures loaded on demand; description as the trigger
- Explicit (slash command) vs implicit (description match) invocation
- When to write a skill (repeated corrections, multi-step procedures, team conventions)
- Hooks: deterministic code on events (pre-tool, post-tool, session start/end)
- Skills guide the model; hooks guarantee behavior regardless of the model

## 4.10 Permissions & Human-in-the-Loop
- Harness permissions vs backend authorization — two layers, both enforced
- Classify tools by effect: read / write / destructive-or-external
- Default-deny for unclassified tools; re-evaluate when the tool list changes
- Argument-level rules (folder scope, amount limits)
- Approval workflows: show arguments, wait for explicit yes
- Escalation paths and audit trails for approved actions

## 4.11 MCP (Model Context Protocol)
- Architecture: client (in harness), server (wraps a service), transport (stdio / HTTP)
- Tool discovery, resources, prompts
- **Build:** an MCP server in front of your own REST API
  - Expose read routes first, writes later with approval
  - Per-user OAuth — never a shared admin key
  - Compact outputs, clear descriptions, server-side validation
- Connect the same server to Claude, your own harness, and any MCP client
- Trust: tool descriptions come from the server — only connect servers you trust; treat annotations as hints

## 4.12 Subagents & Orchestration
- Subagents: separate loops with their own context, returning summaries
- Spawning is just a tool the orchestrator calls
- When to delegate: large searches, independent subtasks, context pressure
- Model routing: strong orchestrator, cheaper workers
- Swarms: planner → many workers → judge/reducer; cost and coordination risks
- A2A: agent-to-agent protocol for remote, independently owned agents

## 4.13 Agent Evaluation
- Final-answer evals (reuse 3.10 methods)
- Trajectory evals: right tools? right arguments? wasted steps? loops?
- Tool routing tests: question sets that should hit each tool
- Cost and latency per task as first-class metrics
- Regression suite in CI, same as 3.10

## 4.14 Agent Security
- **Prompt injection:** direct and indirect (via retrieved docs and tool results)
  - Tool output is data, never instructions
  - Instruction hierarchy, input isolation, sensitive actions behind approval
- **Authorization-aware retrieval:** filter by user permissions *before* ranking; multi-tenant isolation
- **Tool security:** least privilege, scoped credentials, sandboxed execution, confirmation for destructive actions
- **Data exfiltration:** never let tool results steer where user data is sent

---

# Phase 5 — AI Security & Safety

**Objective:** Make the system auditable, compliant, and safe beyond the agent layer.

## 5.1 Input Sanitization
- Strip control characters, validate structure, escape user content in prompts

## 5.2 Output Security
- Guardrails, schema validation before returning to clients, content moderation

## 5.3 Audit Logging
- Log every prompt, retrieval, tool call, approval, and response with authenticated identity
- Immutable trails, retention policies
- Who, what, when, which documents, what was generated

## 5.4 Secrets Management
- API key management, rotation, environment isolation, Vault / cloud secret managers
- MCP servers hold credentials; models never see them

## 5.5 AI Safety
- Jailbreak patterns and defenses, unsafe output categories, moderation APIs, custom classifiers

---

# Phase 6 — Production AI Engineering

**Objective:** Survive real users at scale.

## 6.1 Async Pipelines
- Ingestion, embedding, and OCR jobs on queues (Redis, BullMQ / your Go job queue)

## 6.2 Caching
- Provider prompt caching (stable prefixes: system prompt, tools, skills)
- Embedding cache, retrieval cache, response cache, semantic cache

## 6.3 Observability
- Tracing across retrieval, generation, and every agent step
- Prompt and completion logs, token usage per user/tenant/endpoint
- Latency breakdown: retrieval vs generation vs tools
- Tools: LangSmith, Helicone, Langfuse, or custom OpenTelemetry

## 6.4 Cost Optimization
- Context reduction, compact tool outputs, step limits
- Model routing: small models for simple tasks, large for judgment
- Scripts for deterministic work instead of model reasoning

## 6.5 Prompt Lifecycle Management
- Prompts, tool descriptions, and skills in version control
- Staged rollout, rollbacks, eval gate before promotion
- A prompt change is a deployment

## 6.6 A/B Testing
- Traffic splitting between prompt/model variants
- Decide with eval metrics and statistics, not vibes

## 6.7 Structured Failure Handling
- Malformed JSON, tool failures mid-loop, partial results
- Context length exceeded, zero retrieval results
- Timeouts, retry storms, idempotency for write tools

## 6.8 Production Evaluation & Monitoring
- Sample live traffic into offline evals
- Online signals: thumbs, refusals, escalations, step counts
- Drift detection when models, data, or prompts change

## 6.9 Scalability
- Horizontal scaling of embedding and retrieval services
- Queue-based ingestion, concurrency limits on generation

---

# Phase 7 — Prompt & Behavior Engineering

**Objective:** Fine-grained control, applied as needed — most of it is used inside earlier phases.

## 7.1 Prompt Architecture
- Templates, behavioral and output constraints, task decomposition

## 7.2 Advanced Patterns
- Self-consistency, reflection, meta-prompting
- Evaluate each against your evals — many add cost without gains

## 7.3 Hallucination Management
- Causes: missing context, weak retrieval, ambiguous prompts, high temperature
- Mitigation: fix retrieval first, citations, confidence thresholds, constrained generation

---

# Phase 8 — Frameworks

> Learn abstractions only after you've built what they abstract.

- **Vercel AI SDK** — streaming-first, tool use, provider-agnostic; fits Node/Next.js
- **MCP SDKs** (TypeScript, Python, Go) — for servers and clients
- **LangGraph** — explicit-state agent graphs, branching, human approval steps
- **LangChain.js / LlamaIndex** — optional; useful for retrieval plumbing
- **DSPy** — optional; programmatic prompt optimization

---

# Phase 9 — Advanced Model Customization

> Only after earlier phases, and only when evals show prompting + retrieval aren't enough.

## 9.1 Open-Weight Models
- Llama, Qwen, Mistral, Gemma — licenses, capabilities, tool support

## 9.2 Inference
- Ollama, llama.cpp, vLLM
- Quantization: BF16, INT8, 4-bit; GGUF k-quants (Q4_K_M, Q5_K_M)

## 9.3 Fine-Tuning
- LoRA, QLoRA, PEFT
- Data curation is most of the work
- Always compare against a well-prompted baseline using your evals

---

# Capstone Project

## Multi-Tenant Enterprise RAG + Agent Platform

### Stage 1 — Foundation
- Auth + multi-tenant user model, document upload and storage, basic chat

### Stage 2 — Retrieval
- Pre-processing pipeline, chunking, embeddings, pgvector, hybrid retrieval, reranking

### Stage 3 — Response Quality
- Citations, streaming, "I don't know" handling

### Stage 4 — Evaluation (new position — before anything else is added)
- Golden dataset, retrieval + generation metrics, validated LLM judge
- Ablation table for every Phase 3 technique
- Eval suite runs in CI

### Stage 5 — Agent Layer
- Hand-built harness with step limits and verification
- Agentic retrieval as a tool, compared against the fixed pipeline
- Memory (core + retrieved)
- Human approval for write actions
- Trajectory evals

### Stage 6 — MCP Server
- Expose the platform as an MCP server with per-user OAuth
- Read tools first, then approved write tools
- Demo it working from Claude and from your own harness

### Stage 7 — Security
- Authorization-aware retrieval, tenant isolation, injection mitigations, audit logging

### Stage 8 — Production
- Async ingestion, caching (including prompt caching), observability, prompt versioning, deployment

---

# Interview-Ready Cut

If time is short, this subset makes a strong portfolio story:

1. RAG pipeline (done) **with a real eval suite and ablation numbers** (3.10)
2. Hand-built agent loop with 2–3 well-designed tools and step limits (4.1–4.3)
3. Agentic RAG compared against the fixed pipeline, with numbers (4.8)
4. An MCP server over your own API with OAuth and approval for writes (4.10–4.11)
5. Trajectory evals for the agent (4.13)

Everything else can follow while interviewing.

---

# Highest ROI Topics

1. Evaluation (RAG + agent)
2. RAG Engineering
3. Tool Design & Harness / Loop Engineering
4. Hybrid Retrieval & Re-Ranking
5. MCP (building servers, with auth)
6. AI Security (injection, authorization-aware retrieval, tool permissions)
7. Context Engineering
8. Production Engineering (observability, caching, cost)
9. Workflows vs Agents judgment
10. Embeddings (dense + sparse)

---

*Built for: Backend Engineer → GenAI Engineer transition*
*Stack: Go, Node.js/TypeScript, PostgreSQL, Redis, pgvector, MCP*
