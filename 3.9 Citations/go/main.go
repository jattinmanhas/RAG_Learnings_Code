// main.go — runnable demo for Citations.
//
//	go run .
//
// No API key required. The support checker is lexical word-overlap — a
// stand-in for an NLI model or an LLM judge (3.7).
//
// One model answer, crafted to hit every code path: a citation that's both
// real and supported, one that's real but off-topic, one that's real but
// CONTRADICTS its source (the lexical checker's blind spot — flagged
// explicitly), one hallucinated marker, one sentence the model forgot to
// cite that a source clearly backs, and one sentence nothing backs at all.
package main

import (
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// SOURCES — picked up exactly where 3.8's FormatForCitation left off: a
// numbered map of chunks, one already tagged Kind="summary" (an LLM
// paraphrase, so it can be cited but never quoted).
// ---------------------------------------------------------------------------

var sources = map[int]Source{
	1: {ID: 1, Path: "docs/hnsw-tuning.md:L12-L28",
		Text: "To make HNSW search faster without hurting recall, raise efSearch gradually and re-measure recall on a held-out query set."},
	2: {ID: 2, Path: "docs/hnsw-tuning.md:L30-L44",
		Text: "A practical tuning loop: fix M, sweep efSearch over 40, 80, 160, 320 and plot recall at 10 against p95 latency. Stop at the knee of the curve."},
	3: {ID: 3, Path: "docs/pgvector.md:L60-L120", Kind: "summary",
		Text: "IVFFlat builds far faster than HNSW but its recall degrades as the table grows. (key: faster, recall)"},
	4: {ID: 4, Path: "docs/hnsw-basics.md:L1-L9",
		Text: "HNSW is a graph-based index for approximate nearest neighbour search, built from a hierarchy of navigable small-world graphs."},
	5: {ID: 5, Path: "docs/hnsw-tuning.md:L46-L58",
		Text: "Recall is measured against exact brute-force search on the same query set. Never tune efSearch against production traffic without a ground-truth set."},
}

// A model answer with every failure mode baked in, sentence by sentence:
//
//	[1] real source, supports the claim                    -> verified
//	[2] real source, supports the claim                    -> verified
//	[4] real source, but off-topic                         -> unsupported
//	[3] real source, but CONTRADICTS it (shares keywords)  -> lexical false-pass
//	[7] no such source                                     -> missing (hallucinated)
//	(none) a source backs it, model just didn't cite it    -> backfilled
//	(none) nothing backs it                                -> stays uncited
var modelAnswer = strings.Join([]string{
	"Raise efSearch gradually and re-measure recall on a held-out set [1].",
	"A good tuning loop sweeps efSearch over 40, 80, 160 and 320 while watching p95 latency [2].",
	"HNSW works well for time-series forecasting workloads [4].",
	"IVFFlat builds far faster than HNSW and its recall never degrades as the table grows [3].",
	"Turbo mode can also speed up search [7].",
	"Recall should always be measured against exact brute-force search on the same query set.",
	"This project uses TypeScript and Go for all examples.",
}, " ")

func section(title string) {
	line := strings.Repeat("=", 78)
	fmt.Printf("\n%s\n%s\n%s\n", line, title, line)
}

func intsOrDash(ns []int) string {
	if len(ns) == 0 {
		return "-"
	}
	strs := make([]string, len(ns))
	for i, n := range ns {
		strs[i] = fmt.Sprint(n)
	}
	return strings.Join(strs, ",")
}

func main() {
	fmt.Printf("raw model answer:\n%q\n", modelAnswer)

	// --- 1. Citation Parsing -------------------------------------------------
	section("1. CITATION PARSING — sentence + marker pairs")

	parsed := ParseCitedSentences(modelAnswer)
	for _, s := range parsed {
		fmt.Printf("  cites=[%s]  %q\n", intsOrDash(s.Cites), s.Text)
	}

	// --- 2. Citation Verification ---------------------------------------------
	section("2. CITATION VERIFICATION — existence AND support (threshold 0.5)")

	verified := VerifySentences(parsed, sources, LexicalOverlap, 0.5)
	for _, s := range verified {
		bits := []string{}
		if len(s.Verified) > 0 {
			bits = append(bits, fmt.Sprintf("verified=[%s]", intsOrDash(s.Verified)))
		}
		if len(s.Unsupported) > 0 {
			bits = append(bits, fmt.Sprintf("unsupported=[%s]", intsOrDash(s.Unsupported)))
		}
		if len(s.Missing) > 0 {
			bits = append(bits, fmt.Sprintf("missing=[%s]", intsOrDash(s.Missing)))
		}
		label := strings.Join(bits, " ")
		if label == "" {
			label = "(no citations)"
		}
		fmt.Printf("  %s  %q\n", label, s.Text)
	}
	fmt.Println(`
  note: [3] is flagged "verified" by lexical overlap despite the source
  saying recall DEGRADES and the sentence claiming it NEVER degrades —
  contradiction and support share the same keywords. This is the lexical
  checker's blind spot; escalate to 3.7's LLM groundedness check before
  trusting a citation on a claim that matters.`)

	// --- 3. Citation Repair -----------------------------------------------------
	section("3. CITATION REPAIR — flag strategy (keep sentence, surface the caveat)")

	repaired := RepairAnswer(verified, Flag)
	for _, s := range repaired.Sentences {
		note := ""
		if s.Note != "" {
			note = "  ⚠ " + s.Note
		}
		fmt.Printf("  cites=[%s]%s  %q\n", intsOrDash(s.Cites), note, s.Text)
	}
	fmt.Printf("  %d dropped, %d flagged\n", repaired.Dropped, repaired.Flagged)

	fmt.Println("\n  same input, \"drop\" strategy (safest — removes unverifiable claims entirely):")
	strict := RepairAnswer(verified, Drop)
	for _, s := range strict.Sentences {
		fmt.Printf("  cites=[%s]  %q\n", intsOrDash(s.Cites), s.Text)
	}
	fmt.Printf("  %d dropped, %d flagged\n", strict.Dropped, strict.Flagged)

	// --- 4. Retroactive Attribution ----------------------------------------------
	section("4. RETROACTIVE ATTRIBUTION — best-guess citations for uncited sentences")

	backfilled := BackfillAttribution(repaired.Sentences, sources, LexicalOverlap, 0.4)
	for _, s := range backfilled {
		var tag string
		switch {
		case len(s.Inferred) > 0:
			tag = fmt.Sprintf("inferred=[%s]", intsOrDash(s.Inferred))
		case len(s.Cites) > 0:
			tag = fmt.Sprintf("cites=[%s]", intsOrDash(s.Cites))
		default:
			tag = "uncited"
		}
		fmt.Printf("  %s  %q\n", tag, s.Text)
	}
	fmt.Println(`
  note: the brute-force sentence backfills to [5] — a real source the model
  forgot to cite. The TypeScript/Go sentence stays uncited: nothing in the
  source set clears the threshold, and attribution should never invent one.`)

	// --- 5. Citation Rendering -----------------------------------------------------
	section("5. CITATION RENDERING — inline markers + deduplicated reference list")

	text, references := RenderCitedAnswer(backfilled, sources)
	fmt.Println(text)
	fmt.Printf("\nReferences:\n%s\n", references)
	fmt.Println(`
  note: "~" marks a system-inferred citation — the model never claimed it.`)

	// --- 6. The Complete Citation Pipeline -----------------------------------------
	section("6. THE COMPLETE CITATION PIPELINE")

	result := BuildCitedAnswer(modelAnswer, sources, LexicalOverlap, CitationConfig{
		VerifyThreshold:   0.5,
		BackfillThreshold: 0.4,
		RepairStrategy:    Flag,
	})
	for _, line := range result.Trace {
		fmt.Println(line)
	}
	fmt.Printf("\n%s\nFINAL ANSWER\n%s\n", strings.Repeat("-", 78), strings.Repeat("-", 78))
	fmt.Println(result.Text)
	fmt.Printf("\nReferences:\n%s\n", result.References)
}
