// citations.go — Citations for RAG.
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
//  1. Citation Parsing        — split the answer into sentence + marker pairs
//  2. Citation Verification   — existence AND support, not just existence
//  3. Citation Repair         — what to do with markers that fail either check
//  4. Retroactive Attribution — best-effort citations for sentences with none
//  5. Citation Rendering      — inline markers + a deduplicated reference list
//     + the Complete Citation Pipeline that composes all five.
//
// Support checking and similarity are injected functions, so the lexical mock
// in main.go can be swapped for an NLI model or an LLM judge (3.7) without
// touching this file.
package main

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// SHARED TYPES
// ---------------------------------------------------------------------------

// Source is a source chunk, addressed by the same numeric id 3.8's
// FormatForCitation assigned it — this module picks up exactly where that
// one left off.
type Source struct {
	ID   int
	Text string
	Path string // "docs/hnsw-tuning.md:L12-L28"
	Kind string // "excerpt" | "summary" (paraphrase, never quotable — 3.8) | ""
}

// CitedSentence is one sentence of the answer plus the marker numbers it claims.
type CitedSentence struct {
	Text  string // markers stripped
	Cites []int  // in order of first appearance, deduplicated
}

// ===========================================================================
// 1. CITATION PARSING — turning raw model text into sentence/marker pairs
// ===========================================================================
// A model emits citations inline and often in clusters ("...faster [1][3]."):
// parsing has to split on sentence boundaries (not marker boundaries, since
// one sentence can carry several citations) and strip the markers back out so
// the rendered text doesn't end up with "[1][3]" baked into it twice.

var sentenceEnd = regexp.MustCompile(`([.!?])\s+`)
var marker = regexp.MustCompile(`\[(\d+)\]`)
var spaceBeforePunct = regexp.MustCompile(`\s+([.,;:])`)

func splitAnswerSentences(answer string) []string {
	parts := sentenceEnd.Split(answer, -1)
	marks := sentenceEnd.FindAllStringSubmatch(answer, -1)
	out := []string{}
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if i < len(marks) {
			p += marks[i][1] // put the sentence-ending punctuation back
		}
		out = append(out, p)
	}
	return out
}

// ParseCitedSentences splits an answer into sentences, each carrying its own
// claimed markers.
func ParseCitedSentences(answer string) []CitedSentence {
	out := []CitedSentence{}
	for _, raw := range splitAnswerSentences(answer) {
		cites := []int{}
		seen := map[int]bool{}
		for _, m := range marker.FindAllStringSubmatch(raw, -1) {
			var n int
			fmt.Sscanf(m[1], "%d", &n)
			if !seen[n] {
				seen[n] = true
				cites = append(cites, n)
			}
		}
		text := marker.ReplaceAllString(raw, "")
		text = spaceBeforePunct.ReplaceAllString(text, "$1")
		text = strings.Join(strings.Fields(text), " ")
		out = append(out, CitedSentence{Text: text, Cites: cites})
	}
	return out
}

// ===========================================================================
// 2. CITATION VERIFICATION — existence AND support
// ===========================================================================
// 3.8's ResolveCitations only checked existence: does `n` appear in the
// citations map? That catches a hallucinated marker number, but it does NOT
// catch a model citing a REAL source that has nothing to do with the
// sentence it's attached to — the more common and more dangerous failure,
// because it looks correct at a glance.

// SupportChecker returns how much of claim is backed by sourceText, 0..1.
// Swap for an NLI entailment model or an LLM judge (3.7) in production.
type SupportChecker func(claim, sourceText string) float64

var nonWord = regexp.MustCompile(`\W+`)

func contentWords(s string) []string {
	out := []string{}
	for _, w := range nonWord.Split(strings.ToLower(s), -1) {
		if len(w) > 3 { // drop short function words cheaply
			out = append(out, w)
		}
	}
	return out
}

// LexicalOverlap (2a) is a lexical overlap mock — the fraction of the
// claim's content words that appear somewhere in the source. Cheap,
// zero-dependency, and good enough to catch an OFF-TOPIC citation. It is NOT
// good enough to catch a CONTRADICTING one: "recall degrades" and "recall
// never degrades" share every content word. That gap is exactly why 3.7's
// LLM groundedness check exists — use lexical overlap as the free first
// pass, escalate to it for anything high-stakes.
func LexicalOverlap(claim, sourceText string) float64 {
	claimWords := contentWords(claim)
	if len(claimWords) == 0 {
		return 0
	}
	sourceSet := map[string]bool{}
	for _, w := range contentWords(sourceText) {
		sourceSet[w] = true
	}
	hits := 0
	for _, w := range claimWords {
		if sourceSet[w] {
			hits++
		}
	}
	return float64(hits) / float64(len(claimWords))
}

// VerifiedSentence adds verification results to a CitedSentence.
type VerifiedSentence struct {
	CitedSentence
	Verified    []int // real source AND clears the support threshold
	Unsupported []int // real source, but doesn't support this claim
	Missing     []int // no such source at all — a hallucinated marker (3.8)
}

// VerifySentences checks every claimed marker in every sentence against its source.
func VerifySentences(sentences []CitedSentence, sources map[int]Source, check SupportChecker, threshold float64) []VerifiedSentence {
	out := make([]VerifiedSentence, 0, len(sentences))
	for _, s := range sentences {
		v := VerifiedSentence{CitedSentence: s}
		for _, n := range s.Cites {
			src, ok := sources[n]
			if !ok {
				v.Missing = append(v.Missing, n)
				continue
			}
			if check(s.Text, src.Text) >= threshold {
				v.Verified = append(v.Verified, n)
			} else {
				v.Unsupported = append(v.Unsupported, n)
			}
		}
		out = append(out, v)
	}
	return out
}

// ===========================================================================
// 3. CITATION REPAIR — what to do once verification finds a problem
// ===========================================================================
// A missing marker is always worth discarding — it points at nothing. An
// unsupported marker is judgment-call territory: the sentence might still be
// true, just mis-cited. The three strategies trade completeness for safety.

// RepairStrategy picks how bad citations are handled.
type RepairStrategy string

const (
	Strip RepairStrategy = "strip" // drop the bad marker(s), keep the sentence as an uncited claim
	Flag  RepairStrategy = "flag"  // keep the sentence, append a visible caveat
	Drop  RepairStrategy = "drop"  // remove the whole sentence; safest, costs completeness
)

// RepairedSentence is a sentence after repair; Dropped means it was removed entirely.
type RepairedSentence struct {
	Text    string
	Dropped bool
	Cites   []int // only markers that survived verification
	Note    string
}

func repairSentence(vs VerifiedSentence, strategy RepairStrategy) RepairedSentence {
	bad := append(append([]int{}, vs.Missing...), vs.Unsupported...)
	if len(bad) == 0 {
		return RepairedSentence{Text: vs.Text, Cites: vs.Verified}
	}
	switch strategy {
	case Drop:
		return RepairedSentence{Dropped: true, Note: fmt.Sprintf("dropped — bad citation(s) %v", bad)}
	case Flag:
		return RepairedSentence{Text: vs.Text, Cites: vs.Verified, Note: fmt.Sprintf("citation(s) %v failed verification", bad)}
	default: // Strip
		return RepairedSentence{Text: vs.Text, Cites: vs.Verified}
	}
}

// RepairResult summarises what repair did to an answer.
type RepairResult struct {
	Sentences []RepairedSentence
	Dropped   int
	Flagged   int
}

// RepairAnswer repairs every sentence in an answer with one strategy.
func RepairAnswer(sentences []VerifiedSentence, strategy RepairStrategy) RepairResult {
	res := RepairResult{}
	for _, s := range sentences {
		r := repairSentence(s, strategy)
		if r.Dropped {
			res.Dropped++
			continue
		}
		if r.Note != "" {
			res.Flagged++
		}
		res.Sentences = append(res.Sentences, r)
	}
	return res
}

// ===========================================================================
// 4. RETROACTIVE ATTRIBUTION — citations for sentences that have none
// ===========================================================================
// Not every prompt gets an obediently-cited answer back, and repair (3) can
// strip a sentence down to zero citations. Rather than ship an uncited claim,
// find the source it most resembles and attribute it — clearly marked as the
// SYSTEM's best guess, never conflated with something the model claimed.

// Attributed is a sentence with model-claimed citations kept separate from
// system-inferred ones.
type Attributed struct {
	Text     string
	Cites    []int // model-claimed, verified
	Inferred []int // system-attributed — the model never cited these
}

// BackfillAttribution attributes every still-uncited sentence to its
// best-matching source, if any source clears threshold. Sentences that
// already have a citation are left untouched — attribution only fills gaps,
// it never overrides.
func BackfillAttribution(sentences []RepairedSentence, sources map[int]Source, sim SupportChecker, threshold float64) []Attributed {
	ids := make([]int, 0, len(sources)) // sorted for deterministic tie-breaking
	for id := range sources {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	out := make([]Attributed, 0, len(sentences))
	for _, s := range sentences {
		if len(s.Cites) > 0 {
			out = append(out, Attributed{Text: s.Text, Cites: s.Cites})
			continue
		}
		bestID, bestScore := -1, -1.0
		for _, id := range ids {
			if score := sim(s.Text, sources[id].Text); score > bestScore {
				bestID, bestScore = id, score
			}
		}
		a := Attributed{Text: s.Text, Cites: s.Cites}
		if bestID != -1 && bestScore >= threshold {
			a.Inferred = []int{bestID}
		}
		out = append(out, a)
	}
	return out
}

// ===========================================================================
// 5. CITATION RENDERING — inline markers + a deduplicated reference list
// ===========================================================================
// The reader needs two things: which claim came from where, and a clean list
// of the sources actually used (not the whole retrieval set — see 3.8's
// packing). Inferred citations are marked with a trailing `~` so a reader can
// tell "the model said this" from "the system thinks this is where it came
// from" apart at a glance.

// RenderCitedAnswer renders the final annotated answer plus its reference list.
func RenderCitedAnswer(sentences []Attributed, sources map[int]Source) (text string, references string) {
	usedIDs := []int{}
	seen := map[int]bool{}
	push := func(n int) {
		if !seen[n] {
			seen[n] = true
			usedIDs = append(usedIDs, n)
		}
	}

	lines := make([]string, 0, len(sentences))
	for _, s := range sentences {
		for _, n := range s.Cites {
			push(n)
		}
		for _, n := range s.Inferred {
			push(n)
		}
		markers := ""
		for _, n := range s.Cites {
			markers += fmt.Sprintf("[%d]", n)
		}
		for _, n := range s.Inferred {
			markers += fmt.Sprintf("[%d]~", n)
		}
		if markers != "" {
			lines = append(lines, s.Text+" "+markers)
		} else {
			lines = append(lines, s.Text)
		}
	}

	refLines := make([]string, 0, len(usedIDs))
	for _, n := range usedIDs {
		src := sources[n]
		tag := ""
		if src.Kind == "summary" {
			tag = " (summary — paraphrase, do not quote)"
		}
		refLines = append(refLines, fmt.Sprintf("[%d] %s%s", n, src.Path, tag))
	}
	return strings.Join(lines, " "), strings.Join(refLines, "\n")
}

// ===========================================================================
// 6. THE COMPLETE CITATION PIPELINE
// ===========================================================================
// parse → verify (existence + support) → repair (bad citations) →
// backfill (uncited gaps) → render (markers + reference list), tracking a
// coverage score throughout: the fraction of the final answer's sentences
// that carry at least one citation, model-claimed or system-inferred.

// CitationConfig is the knob panel for the pipeline.
type CitationConfig struct {
	VerifyThreshold   float64
	BackfillThreshold float64
	RepairStrategy    RepairStrategy
}

// CitationResult is the finished, cited answer plus everything needed to audit it.
type CitationResult struct {
	Text       string
	References string
	Coverage   float64
	Trace      []string
}

// BuildCitedAnswer runs the whole pipeline for one raw model answer.
func BuildCitedAnswer(rawAnswer string, sources map[int]Source, check SupportChecker, cfg CitationConfig) CitationResult {
	verifyThreshold := cfg.VerifyThreshold
	if verifyThreshold == 0 {
		verifyThreshold = 0.5
	}
	backfillThreshold := cfg.BackfillThreshold
	if backfillThreshold == 0 {
		backfillThreshold = 0.4
	}
	strategy := cfg.RepairStrategy
	if strategy == "" {
		strategy = Flag
	}

	trace := []string{}

	parsed := ParseCitedSentences(rawAnswer)
	trace = append(trace, fmt.Sprintf("parse: %d sentences", len(parsed)))

	verified := VerifySentences(parsed, sources, check, verifyThreshold)
	missingCount, unsupportedCount := 0, 0
	for _, s := range verified {
		missingCount += len(s.Missing)
		unsupportedCount += len(s.Unsupported)
	}
	trace = append(trace, fmt.Sprintf("verify: %d hallucinated marker(s), %d unsupported citation(s)", missingCount, unsupportedCount))

	repairRes := RepairAnswer(verified, strategy)
	trace = append(trace, fmt.Sprintf("repair (%s): %d sentence(s) dropped, %d flagged", strategy, repairRes.Dropped, repairRes.Flagged))

	backfilled := BackfillAttribution(repairRes.Sentences, sources, check, backfillThreshold)
	backfillCount := 0
	for _, s := range backfilled {
		if len(s.Inferred) > 0 {
			backfillCount++
		}
	}
	trace = append(trace, fmt.Sprintf("backfill: %d previously-uncited sentence(s) attributed", backfillCount))

	text, references := RenderCitedAnswer(backfilled, sources)
	cited := 0
	for _, s := range backfilled {
		if len(s.Cites) > 0 || len(s.Inferred) > 0 {
			cited++
		}
	}
	coverage := 0.0
	if len(backfilled) > 0 {
		coverage = float64(cited) / float64(len(backfilled))
	}
	trace = append(trace, fmt.Sprintf("coverage: %d/%d sentences carry a citation (%.0f%%)", cited, len(backfilled), coverage*100))

	return CitationResult{Text: text, References: references, Coverage: coverage, Trace: trace}
}
