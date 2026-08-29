// index.ts — runnable demo for Citations.
//
//   npm start
//
// No API key required. The support checker is lexical word-overlap — a
// stand-in for an NLI model or an LLM judge (3.7).
//
// One model answer, crafted to hit every code path: a citation that's both
// real and supported, one that's real but off-topic, one that's real but
// CONTRADICTS its source (the lexical checker's blind spot — flagged
// explicitly), one hallucinated marker, one sentence the model forgot to
// cite that a source clearly backs, and one sentence nothing backs at all.

import {
  Source,
  parseCitedSentences,
  lexicalOverlap,
  verifySentences,
  repairAnswer,
  backfillAttribution,
  renderCitedAnswer,
  buildCitedAnswer,
} from "./citations";

// ---------------------------------------------------------------------------
// SOURCES — picked up exactly where 3.8's formatForCitation left off: a
// numbered map of chunks, one already tagged kind="summary" (an LLM
// paraphrase, so it can be cited but never quoted).
// ---------------------------------------------------------------------------

const SOURCES = new Map<number, Source>(
  [
    {
      id: 1,
      path: "docs/hnsw-tuning.md:L12-L28",
      text: "To make HNSW search faster without hurting recall, raise efSearch gradually and re-measure recall on a held-out query set.",
    },
    {
      id: 2,
      path: "docs/hnsw-tuning.md:L30-L44",
      text: "A practical tuning loop: fix M, sweep efSearch over 40, 80, 160, 320 and plot recall at 10 against p95 latency. Stop at the knee of the curve.",
    },
    {
      id: 3,
      path: "docs/pgvector.md:L60-L120",
      kind: "summary",
      text: "IVFFlat builds far faster than HNSW but its recall degrades as the table grows. (key: faster, recall)",
    },
    {
      id: 4,
      path: "docs/hnsw-basics.md:L1-L9",
      text: "HNSW is a graph-based index for approximate nearest neighbour search, built from a hierarchy of navigable small-world graphs.",
    },
    {
      id: 5,
      path: "docs/hnsw-tuning.md:L46-L58",
      text: "Recall is measured against exact brute-force search on the same query set. Never tune efSearch against production traffic without a ground-truth set.",
    },
  ].map((s) => [s.id, s as Source])
);

// A model answer with every failure mode baked in, sentence by sentence:
//   [1] real source, supports the claim               -> verified
//   [2] real source, supports the claim                -> verified
//   [4] real source, but off-topic                      -> unsupported
//   [3] real source, but CONTRADICTS it (shares keywords)-> lexical false-pass
//   [7] no such source                                  -> missing (hallucinated)
//   (none) a source backs it, model just didn't cite it -> backfilled
//   (none) nothing backs it                              -> stays uncited
const MODEL_ANSWER = [
  "Raise efSearch gradually and re-measure recall on a held-out set [1].",
  "A good tuning loop sweeps efSearch over 40, 80, 160 and 320 while watching p95 latency [2].",
  "HNSW works well for time-series forecasting workloads [4].",
  "IVFFlat builds far faster than HNSW and its recall never degrades as the table grows [3].",
  "Turbo mode can also speed up search [7].",
  "Recall should always be measured against exact brute-force search on the same query set.",
  "This project uses TypeScript and Go for all examples.",
].join(" ");

const section = (title: string) => console.log(`\n${"=".repeat(78)}\n${title}\n${"=".repeat(78)}`);

console.log(`raw model answer:\n"${MODEL_ANSWER}"`);

// --- 1. Citation Parsing -------------------------------------------------------
section("1. CITATION PARSING — sentence + marker pairs");

const parsed = parseCitedSentences(MODEL_ANSWER);
for (const s of parsed) {
  console.log(`  cites=[${s.cites.join(",") || "-"}]  "${s.text}"`);
}

// --- 2. Citation Verification ---------------------------------------------------
section("2. CITATION VERIFICATION — existence AND support (threshold 0.5)");

const verified = verifySentences(parsed, SOURCES, lexicalOverlap, 0.5);
for (const s of verified) {
  const bits = [
    s.verified.length ? `verified=[${s.verified}]` : "",
    s.unsupported.length ? `unsupported=[${s.unsupported}]` : "",
    s.missing.length ? `missing=[${s.missing}]` : "",
  ]
    .filter(Boolean)
    .join(" ");
  console.log(`  ${bits || "(no citations)"}  "${s.text}"`);
}
console.log(`
  note: [3] is flagged "verified" by lexical overlap despite the source
  saying recall DEGRADES and the sentence claiming it NEVER loses recall —
  contradiction and support share the same keywords. This is the lexical
  checker's blind spot; escalate to 3.7's LLM groundedness check before
  trusting a citation on a claim that matters.`);

// --- 3. Citation Repair ---------------------------------------------------------
section("3. CITATION REPAIR — flag strategy (keep sentence, surface the caveat)");

const { sentences: repaired, dropped, flagged } = repairAnswer(verified, "flag");
for (const s of repaired) {
  console.log(`  cites=[${s.cites.join(",") || "-"}]${s.note ? `  ⚠ ${s.note}` : ""}  "${s.text}"`);
}
console.log(`  ${dropped} dropped, ${flagged} flagged`);

console.log(`\n  same input, "drop" strategy (safest — removes unverifiable claims entirely):`);
const strict = repairAnswer(verified, "drop");
for (const s of strict.sentences) console.log(`  cites=[${s.cites.join(",") || "-"}]  "${s.text}"`);
console.log(`  ${strict.dropped} dropped, ${strict.flagged} flagged`);

// --- 4. Retroactive Attribution --------------------------------------------------
section("4. RETROACTIVE ATTRIBUTION — best-guess citations for uncited sentences");

const backfilled = backfillAttribution(repaired, SOURCES, lexicalOverlap, 0.4);
for (const s of backfilled) {
  const tag = s.inferred.length ? `inferred=[${s.inferred}]` : s.cites.length ? `cites=[${s.cites}]` : "uncited";
  console.log(`  ${tag}  "${s.text}"`);
}
console.log(`
  note: the brute-force sentence backfills to [5] — a real source the model
  forgot to cite. The TypeScript/Go sentence stays uncited: nothing in the
  source set clears the threshold, and attribution should never invent one.`);

// --- 5. Citation Rendering -------------------------------------------------------
section("5. CITATION RENDERING — inline markers + deduplicated reference list");

const rendered = renderCitedAnswer(backfilled, SOURCES);
console.log(rendered.text);
console.log(`\nReferences:\n${rendered.references}`);
console.log(`\n  note: "~" marks a system-inferred citation — the model never claimed it.`);

// --- 6. The Complete Citation Pipeline --------------------------------------------
section("6. THE COMPLETE CITATION PIPELINE");

const result = buildCitedAnswer(MODEL_ANSWER, SOURCES, lexicalOverlap, {
  verifyThreshold: 0.5,
  backfillThreshold: 0.4,
  repairStrategy: "flag",
});
for (const line of result.trace) console.log(line);
console.log(`\n${"-".repeat(78)}\nFINAL ANSWER\n${"-".repeat(78)}`);
console.log(result.text);
console.log(`\nReferences:\n${result.references}`);
