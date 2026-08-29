// citations.ts — Citations for RAG.
//
// 3.8 got you as far as a model answer sprinkled with `[n]` markers and a
// map from `n` back to the chunk that produced it. Two things that map does
// NOT tell you:
//
//   - does marker `[n]` point at a REAL source at all?        (existence)
//   - if it does, does that source actually SUPPORT the claim  (support)
//     the model attached it to, or did the model just grab the
//     nearest number?
//
// This file is the pipeline that turns a raw, possibly-wrong answer into one
// you can show a user with confidence:
//
//   1. Citation Parsing       — split the answer into sentence + marker pairs
//   2. Citation Verification  — existence AND support, not just existence
//   3. Citation Repair        — what to do with markers that fail either check
//   4. Retroactive Attribution— best-effort citations for sentences with none
//   5. Citation Rendering     — inline markers + a deduplicated reference list
//   + the Complete Citation Pipeline that composes all five.
//
// Support checking and similarity are injected functions, so the lexical mock
// in index.ts can be swapped for an NLI model or an LLM judge (3.7) without
// touching this file.

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

/** A source chunk, addressed by the same numeric id 3.8's formatForCitation
 *  assigned it — this module picks up exactly where that one left off. */
export interface Source {
  id: number;
  text: string;
  path: string; // "docs/hnsw-tuning.md:L12-L28"
  kind?: "excerpt" | "summary"; // summary = LLM paraphrase, never quotable (3.8)
}

/** One sentence of the answer plus the marker numbers it claims. */
export interface CitedSentence {
  text: string; // markers stripped
  cites: number[]; // in order of first appearance, deduplicated
}

// ===========================================================================
// 1. CITATION PARSING — turning raw model text into sentence/marker pairs
// ===========================================================================
// A model emits citations inline and often in clusters ("...faster [1][3].").
// Parsing has to (a) split on sentence boundaries, not marker boundaries —
// one sentence can carry several citations — and (b) strip the markers back
// out so the rendered text doesn't end up with "[1][3]" baked into it twice.

const SENTENCE_SPLIT = /(?<=[.!?])\s+/;
const MARKER = /\[(\d+)\]/g;

/** Split an answer into sentences, each carrying its own claimed markers. */
export function parseCitedSentences(answer: string): CitedSentence[] {
  return answer
    .split(SENTENCE_SPLIT)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((raw) => {
      const cites: number[] = [];
      const seen = new Set<number>();
      for (const m of raw.matchAll(MARKER)) {
        const n = Number(m[1]);
        if (!seen.has(n)) {
          seen.add(n);
          cites.push(n);
        }
      }
      const text = raw
        .replace(MARKER, "")
        .replace(/\s+([.,;:])/g, "$1") // markers leave a space before punctuation
        .replace(/\s+/g, " ")
        .trim();
      return { text, cites };
    });
}

// ===========================================================================
// 2. CITATION VERIFICATION — existence AND support
// ===========================================================================
// 3.8's resolveCitations() only checked existence: does `n` appear in the
// citations map? That catches a hallucinated marker number, but it does NOT
// catch a model citing a REAL source that has nothing to do with the
// sentence it's attached to — the more common and more dangerous failure,
// because it looks correct at a glance.

/** Returns how much of `claim` is backed by `sourceText`, 0..1.
 *  Swap for an NLI entailment model or an LLM judge (3.7) in production. */
export type SupportChecker = (claim: string, sourceText: string) => number;

function contentWords(s: string): string[] {
  return s
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 3); // drop short function words cheaply
}

/** 2a. Lexical overlap mock — fraction of the claim's content words that
 *  appear somewhere in the source. Cheap, zero-dependency, and good enough
 *  to catch an OFF-TOPIC citation. It is NOT good enough to catch a
 *  CONTRADICTING one: "recall degrades" and "recall never degrades" share
 *  every content word. That gap is exactly why 3.7's LLM groundedness check
 *  exists — use lexical overlap as the free first pass, escalate to it for
 *  anything high-stakes. */
export const lexicalOverlap: SupportChecker = (claim, sourceText) => {
  const claimWords = contentWords(claim);
  if (claimWords.length === 0) return 0;
  const sourceSet = new Set(contentWords(sourceText));
  const hits = claimWords.filter((w) => sourceSet.has(w)).length;
  return hits / claimWords.length;
};

export interface VerifiedSentence extends CitedSentence {
  verified: number[]; // real source AND clears the support threshold
  unsupported: number[]; // real source, but doesn't support this claim
  missing: number[]; // no such source at all — a hallucinated marker (3.8)
}

/** Check every claimed marker in every sentence against its source. */
export function verifySentences(
  sentences: CitedSentence[],
  sources: Map<number, Source>,
  check: SupportChecker,
  threshold = 0.5
): VerifiedSentence[] {
  return sentences.map((s) => {
    const verified: number[] = [];
    const unsupported: number[] = [];
    const missing: number[] = [];
    for (const n of s.cites) {
      const src = sources.get(n);
      if (!src) {
        missing.push(n);
        continue;
      }
      const score = check(s.text, src.text);
      (score >= threshold ? verified : unsupported).push(n);
    }
    return { ...s, verified, unsupported, missing };
  });
}

// ===========================================================================
// 3. CITATION REPAIR — what to do once verification finds a problem
// ===========================================================================
// A missing marker is always worth discarding — it points at nothing. An
// unsupported marker is judgment call territory: the sentence might still be
// true, just mis-cited. The three strategies trade completeness for safety.

export type RepairStrategy = "strip" | "flag" | "drop";
// strip — drop the bad marker(s), keep the sentence text as an uncited claim
// flag  — keep the sentence, append a visible caveat for the reader/caller
// drop  — remove the whole sentence; safest, costs completeness

export interface RepairedSentence {
  text: string | null; // null = sentence dropped entirely
  cites: number[]; // only markers that survived verification
  note?: string;
}

function repairSentence(vs: VerifiedSentence, strategy: RepairStrategy): RepairedSentence {
  const bad = [...vs.missing, ...vs.unsupported];
  if (bad.length === 0) return { text: vs.text, cites: vs.verified };

  if (strategy === "drop") {
    return { text: null, cites: [], note: `dropped — bad citation(s) [${bad.join(",")}]` };
  }
  if (strategy === "flag") {
    return {
      text: vs.text,
      cites: vs.verified,
      note: `citation(s) [${bad.join(",")}] failed verification`,
    };
  }
  return { text: vs.text, cites: vs.verified }; // strip
}

/** Repair every sentence in an answer with one strategy. */
export function repairAnswer(
  sentences: VerifiedSentence[],
  strategy: RepairStrategy
): { sentences: { text: string; cites: number[]; note?: string }[]; dropped: number; flagged: number } {
  const repaired = sentences.map((s) => repairSentence(s, strategy));
  return {
    sentences: repaired.filter((s): s is { text: string; cites: number[]; note?: string } => s.text !== null),
    dropped: repaired.filter((s) => s.text === null).length,
    flagged: repaired.filter((s) => s.text !== null && s.note !== undefined).length,
  };
}

// ===========================================================================
// 4. RETROACTIVE ATTRIBUTION — citations for sentences that have none
// ===========================================================================
// Not every prompt gets an obediently-cited answer back, and repair (3) can
// strip a sentence down to zero citations. Rather than ship an uncited claim,
// find the source it most resembles and attribute it — clearly marked as the
// SYSTEM's best guess, never conflated with something the model claimed.

export interface Attributed {
  text: string;
  cites: number[]; // model-claimed, verified
  inferred: number[]; // system-attributed — the model never cited these
}

/** Attribute every still-uncited sentence to its best-matching source, if
 *  any source clears `threshold`. Sentences that already have a citation are
 *  left untouched — attribution only fills gaps, it never overrides. */
export function backfillAttribution(
  sentences: { text: string; cites: number[] }[],
  sources: Map<number, Source>,
  sim: SupportChecker,
  threshold = 0.4
): Attributed[] {
  return sentences.map((s) => {
    if (s.cites.length > 0) return { ...s, inferred: [] };
    let best: { id: number; score: number } | null = null;
    for (const src of sources.values()) {
      const score = sim(s.text, src.text);
      if (!best || score > best.score) best = { id: src.id, score };
    }
    return { ...s, inferred: best && best.score >= threshold ? [best.id] : [] };
  });
}

// ===========================================================================
// 5. CITATION RENDERING — inline markers + a deduplicated reference list
// ===========================================================================
// The reader needs two things: which claim came from where, and a clean list
// of the sources actually used (not the whole retrieval set — see 3.8's
// packing). Inferred citations are marked with a trailing `~` so a reader can
// tell "the model said this" from "the system thinks this is where it came
// from" apart at a glance.

export function renderCitedAnswer(
  sentences: Attributed[],
  sources: Map<number, Source>
): { text: string; references: string } {
  const usedIds: number[] = [];
  const pushUsed = (n: number) => {
    if (!usedIds.includes(n)) usedIds.push(n);
  };

  const lines = sentences.map((s) => {
    s.cites.forEach(pushUsed);
    s.inferred.forEach(pushUsed);
    const markers = [...s.cites.map((n) => `[${n}]`), ...s.inferred.map((n) => `[${n}]~`)].join("");
    return markers ? `${s.text} ${markers}` : s.text;
  });

  const references = usedIds
    .map((n) => {
      const src = sources.get(n)!;
      const tag = src.kind === "summary" ? " (summary — paraphrase, do not quote)" : "";
      return `[${n}] ${src.path}${tag}`;
    })
    .join("\n");

  return { text: lines.join(" "), references };
}

// ===========================================================================
// 6. THE COMPLETE CITATION PIPELINE
// ===========================================================================
// parse → verify (existence + support) → repair (bad citations) →
// backfill (uncited gaps) → render (markers + reference list), tracking a
// coverage score throughout: the fraction of the final answer's sentences
// that carry at least one citation, model-claimed or system-inferred.

export interface CitationConfig {
  verifyThreshold?: number;
  backfillThreshold?: number;
  repairStrategy?: RepairStrategy;
}

export interface CitationResult {
  text: string;
  references: string;
  coverage: number;
  trace: string[];
}

export function buildCitedAnswer(
  rawAnswer: string,
  sources: Map<number, Source>,
  check: SupportChecker,
  cfg: CitationConfig = {}
): CitationResult {
  const verifyThreshold = cfg.verifyThreshold ?? 0.5;
  const backfillThreshold = cfg.backfillThreshold ?? 0.4;
  const strategy = cfg.repairStrategy ?? "flag";
  const trace: string[] = [];

  const parsed = parseCitedSentences(rawAnswer);
  trace.push(`parse: ${parsed.length} sentences`);

  const verified = verifySentences(parsed, sources, check, verifyThreshold);
  const missingCount = verified.reduce((n, s) => n + s.missing.length, 0);
  const unsupportedCount = verified.reduce((n, s) => n + s.unsupported.length, 0);
  trace.push(`verify: ${missingCount} hallucinated marker(s), ${unsupportedCount} unsupported citation(s)`);

  const { sentences: repaired, dropped, flagged } = repairAnswer(verified, strategy);
  trace.push(`repair (${strategy}): ${dropped} sentence(s) dropped, ${flagged} flagged`);

  const backfilled = backfillAttribution(repaired, sources, check, backfillThreshold);
  const backfillCount = backfilled.filter((s) => s.inferred.length > 0).length;
  trace.push(`backfill: ${backfillCount} previously-uncited sentence(s) attributed`);

  const { text, references } = renderCitedAnswer(backfilled, sources);
  const cited = backfilled.filter((s) => s.cites.length > 0 || s.inferred.length > 0).length;
  const coverage = backfilled.length === 0 ? 0 : cited / backfilled.length;
  trace.push(
    `coverage: ${cited}/${backfilled.length} sentences carry a citation (${(coverage * 100).toFixed(0)}%)`
  );

  return { text, references, coverage, trace };
}
