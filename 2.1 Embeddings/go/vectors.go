// vectors.go — the vector math you actually need in a RAG pipeline.
package main

import "math"

// CosineSimilarity is the standard measure of semantic closeness between two
// embeddings: 1.0 = identical direction, 0 = orthogonal, -1 = opposite.
//
// If your model already returns L2-normalized (unit-length) vectors — as
// nomic-embed-text:v1.5 does — the denominator is always 1 and a plain dot
// product gives the same answer more cheaply (see DotProduct). Computing the
// full cosine is the safe default that stays correct for ANY model.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// DotProduct equals cosine similarity ONLY when both vectors are unit-length.
// Vector databases store normalized vectors precisely so they can use this
// cheaper operation at query time.
func DotProduct(a, b []float64) float64 {
	var dot float64
	for i := range a {
		if i >= len(b) {
			break
		}
		dot += a[i] * b[i]
	}
	return dot
}

// Normalize scales a vector to unit length in place-safe fashion (returns a new
// slice). Do this once at ingest if your model doesn't normalize for you, then
// you can rely on DotProduct everywhere downstream.
func Normalize(v []float64) []float64 {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// TruncateDims implements Matryoshka-style dimension reduction. Models like
// nomic-embed-text and text-embedding-3 are trained so the first N dimensions
// still carry most of the signal. Truncating (then re-normalizing) shrinks
// storage and speeds up search at a small accuracy cost — a real production
// lever when you have millions of vectors.
func TruncateDims(v []float64, dims int) []float64 {
	if dims >= len(v) {
		return v
	}
	return Normalize(v[:dims])
}
