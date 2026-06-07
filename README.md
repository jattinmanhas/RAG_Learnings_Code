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

### Understand
- How chunk size affects retrieval precision vs recall
- Why chunk boundaries matter for answer completeness

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

### Learn
- Top-K Retrieval
- Metadata Filtering
- Hybrid Retrieval
- Retrieval Tuning

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

### Understand
- Why two-stage retrieval is the production standard
- The latency/accuracy tradeoff between stages

---

## 3.7 Retrieval Confidence & "I Don't Know" Handling

### Learn
- Retrieval score thresholding — if no chunk exceeds a similarity threshold, don't answer
- Explicit abstention — model says "I don't have information about this" rather than hallucinating
- Confidence signals — using retrieval scores as a proxy for answer confidence

### Understand
- Hallucination often happens when retrieval fails silently
- Explicit "no result" paths reduce hallucination rate significantly

---

## 3.8 Context Construction

### Learn
- Context Packing — fitting maximum relevant content within token limits
- Context Ordering — highest relevance at top and bottom, not in the middle (lost-in-the-middle)
- Context Compression — summarizing or trimming chunks before injection

---

## 3.9 Citations

### Learn
- Source tracking through the pipeline
- Chunk attribution in the response
- Grounded responses with references

---

## 3.10 RAG Evaluation

### Learn
- Retrieval Precision
- Retrieval Recall
- Answer Relevance
- Faithfulness
- Hallucination Rate

---

# Phase 4 — AI Behavior Engineering

**Objective:** Control model behavior intentionally and systematically.

> This is the phase most GenAI roadmaps skip entirely.

---

## 4.1 Prompt Architecture

### Learn
- Prompt Templates
- Behavioral Constraints
- Output Constraints
- Task Decomposition

### Advanced Prompting Patterns
- Self-Consistency — run CoT multiple times, take majority answer
- Reflection Prompting — ask the model to review its own output
- Meta-Prompting — use a model to generate or improve prompts

---

## 4.2 Context Engineering

### Context Construction
Combine at inference time:
- User input
- Retrieved chunks
- Memory
- Tool outputs
- System state

### Context Compression
- Summarization of older conversation turns
- Memory pruning — remove low-signal history
- Rolling context — maintain a compressed summary + recent verbatim turns

### Token Budget Management
- Actively decide what gets in and what gets dropped
- Prioritize by recency, relevance, and instruction-criticality
- System prompt and current query are never dropped

---

## 4.3 Hallucination Management

### Causes
- Missing context (retrieval didn't find the right chunk)
- Weak retrieval (wrong chunks retrieved)
- Ambiguous prompts (model fills gaps with invented content)
- Temperature too high

### Mitigation
- Improve retrieval before blaming the model
- Explicit citations in every answer
- Retrieval confidence thresholding
- Constrained generation (only answer from provided context)

---

## 4.4 Memory Systems

### Short-Term Memory
Chat history within a session.

### Long-Term Memory
Persistent memory across sessions — user preferences, facts, history.

### Episodic Memory
Previous agent actions, decisions, and their outcomes.

---

## 4.5 Reasoning Systems

### ReAct
```
Reason → Act → Observe → Reason → ...
```

### Reflection
Self-review loops — model critiques its own output and revises.

### Planning
Task decomposition — break a complex goal into ordered sub-tasks before execution.

---

# Phase 5 — Agents & Tool Use

**Objective:** Build systems that can act, not just respond.

---

## 5.1 Function Calling

### Learn
- Tool schema definition
- Tool invocation
- Tool selection logic
- Tool output handling
- Tool validation

### Examples
- Search APIs
- Database queries
- External APIs
- Calculators
- Code executors

---

## 5.2 Tool Orchestration

### Learn
- Sequential tool execution
- Tool chaining — output of one tool as input to next
- Fallback logic — what happens when a tool fails

---

## 5.3 Parallel Tool Calling

### Learn
- Concurrent tool execution — model calls multiple tools in a single turn
- Latency optimization through parallelism
- Handling partial failures in parallel calls

---

## 5.4 Human-in-the-Loop

### Learn
- Approval workflows — agent pauses and requests human confirmation before high-risk actions
- Action classification — distinguish read-only actions from destructive/irreversible ones
- Escalation paths — when agent uncertainty exceeds threshold, route to human
- Audit trails for human-approved decisions

### Understand
- Not all agent actions should be autonomous
- Production agents on real data require trust boundaries
- HITL is a safety mechanism, not a failure of automation

---

## 5.5 Agent Workflows

### Build
- Research Agents
- Coding Agents
- Knowledge Agents
- Workflow Automation Agents

---

## 5.6 MCP (Model Context Protocol)

### Learn
- MCP Architecture
- Client/Server model
- Tool discovery
- Resource exposure
- Authentication in MCP

### Understand
- Why MCP is becoming the standard interface between agents and external systems
- How it compares to ad-hoc function calling

---

# Phase 6 — AI Security & Safety

**Objective:** Build AI systems that are secure, compliant, and trustworthy.

> Given your background in Password Managers and IAM, this phase is a significant differentiator.

---

## 6.1 Prompt Injection

### Direct Injection
User inputs instructions designed to override system prompt behavior.

### Indirect Injection
Malicious instructions embedded in retrieved documents — the model executes them during generation.

### Learn
- Detection patterns
- Mitigation strategies — instruction hierarchy, input sanitization
- Why RAG systems are particularly vulnerable to indirect injection

---

## 6.2 Input Sanitization

### Learn
- Stripping control characters and injection patterns from user input
- Validating input structure before sending to model
- Escaping user content when embedding in system prompts

---

## 6.3 RAG Security

### Learn
- Data leakage prevention — model should not reveal chunks from documents the user is unauthorized to see
- Multi-tenant isolation — tenant A's documents must never surface in tenant B's retrieval
- Authorization-aware retrieval — filter retrieval candidates by user permissions before ranking
- Document-level permission enforcement

---

## 6.4 Tool Security

### Learn
- Permission validation before tool execution
- Action confirmation for destructive operations
- Least privilege design — tools get minimum necessary permissions
- Sandboxing tool execution environments

---

## 6.5 Output Security

### Learn
- Guardrails — blocking or flagging unsafe outputs
- Output validation against schema before returning to client
- Content moderation integration

---

## 6.6 Audit Logging

### Learn
- Logging every prompt, retrieval event, tool call, and response with user identity
- Immutable audit trails for compliance
- Log structure — who, what, when, which documents were retrieved, what was generated
- Retention policies

### Understand
- Enterprise AI procurement requires auditability
- Audit logs are the evidence layer for investigating security incidents
- Directly connected to IAM — logs must carry authenticated identity, not just session IDs

---

## 6.7 Secrets Management

### Learn
- API key management
- Secret rotation
- Environment isolation
- Vault concepts (HashiCorp Vault, AWS Secrets Manager)

---

## 6.8 AI Safety

### Learn
- Jailbreak patterns and defenses
- Unsafe output categories
- Moderation APIs and custom classifiers

---

# Phase 7 — Production AI Engineering

**Objective:** Build AI systems that survive real users at scale.

---

## 7.1 Async Pipelines

### Build
- Document ingestion pipelines
- Embedding generation jobs
- OCR pipelines for scanned documents

### Use
- Redis
- BullMQ

---

## 7.2 Caching

### Learn
- Embedding cache — don't re-embed identical content
- Retrieval cache — cache results for repeated queries
- Response cache — cache full responses for identical prompt + context pairs
- Semantic caching — cache responses for semantically similar queries

---

## 7.3 Observability

### Learn
- Distributed tracing across the full RAG pipeline
- Prompt logging — store every prompt and completion
- Token monitoring — track usage per user, per tenant, per endpoint
- Latency analysis — identify bottlenecks in retrieval vs generation
- Tools: LangSmith, Helicone, custom solutions

---

## 7.4 Cost Optimization

### Learn
- Context reduction — trim unnecessary tokens before sending
- Caching strategies — avoid redundant API calls
- Model routing — use smaller/cheaper models for simple tasks, larger models for complex ones
- Prompt compression — reduce prompt length without losing instruction quality

---

## 7.5 Prompt Lifecycle Management

### Learn
- Prompt versioning — track prompts in version control like code
- Staged rollout — test prompt changes in staging before production
- Rollbacks — revert to previous prompt version on regression
- Prompt evaluation against regression test sets before promoting

> Treat prompts like code. A prompt change is a deployment.

---

## 7.6 A/B Testing Prompts

### Learn
- Running multiple prompt versions simultaneously in production
- Traffic splitting between variants
- Measuring performance with evaluation metrics, not just vibes
- Deciding when a variant is statistically better before full rollout

### Understand
- Prompt changes can silently degrade quality without A/B infrastructure
- This is how responsible prompt changes are shipped in production

---

## 7.7 Structured Failure Handling

### Handle
- Malformed JSON output despite JSON mode
- Tool call failures mid-chain
- Partial tool responses
- Context length exceeded at runtime
- Retrieval returning zero results
- Model timeout and retry storms

### Understand
- LLM failure modes are different from normal API failures
- Retry logic for LLMs must account for idempotency and cost

---

## 7.8 AI Evaluation

### Learn
- Faithfulness — is the answer consistent with the retrieved context?
- Groundedness — is every claim traceable to a source?
- Answer Relevance — does the answer actually address the query?
- Hallucination Rate — what percentage of claims are fabricated?
- Tools: RAGAS, DeepEval

---

## 7.9 Scalability

### Learn
- Horizontal scaling of embedding and retrieval services
- Queue-based architectures for ingestion workloads
- Load management under concurrent generation requests

---

# Phase 8 — Frameworks

**Objective:** Learn abstractions only after understanding what they abstract.

> Do not start here. These frameworks are easier to evaluate, debug, and extend once you've built the underlying patterns manually.

---

## 8.1 LangChain.js

- Chains
- Agents
- Tools
- Retrieval

---

## 8.2 Vercel AI SDK

- Streaming-first design for Node.js / Next.js
- Built-in tool use
- Provider-agnostic (OpenAI, Anthropic, etc.)
- Production-friendly for JS backends

---

## 8.3 LlamaIndex

- Indexes
- Retrieval pipelines
- RAG abstractions

---

## 8.4 DSPy (Optional)

- Programmatic prompt optimization
- Automatic few-shot generation
- Signature-based prompting

---

# Phase 9 — Advanced Model Customization

**Objective:** Understand how models are adapted for specific domains.

> Only after completing all earlier phases.

---

## 9.1 Fine-Tuning

- LoRA
- QLoRA
- PEFT

---

## 9.2 Open-Source Models

- Llama
- Qwen
- Mistral

---

## 9.3 Inference

- vLLM
- Ollama
- Quantization

---

# Capstone Project

## Multi-Tenant Enterprise RAG Platform

A single project, built progressively across all phases.

---

### Stage 1 — Foundation
- Authentication + multi-tenant user model
- PDF upload + document storage
- Basic chat with an LLM

### Stage 2 — Retrieval
- Document pre-processing pipeline
- Chunking + embedding generation
- pgvector integration
- Semantic retrieval

### Stage 3 — Response Quality
- Citations in every response
- Streaming responses
- "I don't know" handling when retrieval fails

### Stage 4 — Memory
- Short-term chat memory
- Long-term user memory
- Context compression for long conversations

### Stage 5 — Tool Use
- Function calling
- At least one real tool (web search, database query, calculator)
- Human-in-the-loop approval for destructive actions

### Stage 6 — Security
- Authorization-aware retrieval (users can only retrieve their own documents)
- Multi-tenant isolation
- Audit logging for all model interactions
- Prompt injection mitigation

### Stage 7 — Production
- Async ingestion pipeline (BullMQ + Redis)
- Caching layer
- Observability (tracing, prompt logs, token tracking)
- Prompt versioning
- AI evaluation suite (faithfulness, hallucination rate)
- Production deployment

---

# Highest ROI Topics

If you had to prioritize, these topics will take you further than 90% of people currently putting "GenAI" on their resumes:

1. RAG Engineering
2. Embeddings (dense + sparse)
3. pgvector
4. Query Transformation
5. Re-Ranking
6. Function Calling
7. AI Security
8. AI Evaluation
9. Production Engineering
10. MCP

---

*Built for: Backend Engineer → GenAI Engineer transition*
*Stack: Node.js, Go, PostgreSQL, Redis, pgvector*
