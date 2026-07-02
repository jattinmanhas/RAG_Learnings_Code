// main.go — runnable demo of production embedding patterns.
//
//	cd "2.1 Embeddings/go" && go run .
//
// Requires a local Ollama with the embedding model pulled:
//
//	ollama pull nomic-embed-text:v1.5
//
// Point baseURL/apiKey in client.go at OpenAI to run against a hosted model.
package main

import (
	"context"
	"fmt"
	"log"
	"time"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient()

	// ------------------------------------------------------------------
	// 1. Single embedding — inspect shape and dimensionality.
	// ------------------------------------------------------------------
	vecs, err := client.Embed(ctx, []string{"The quick brown fox jumps over the lazy dog."})
	if err != nil {
		log.Fatalf("embed failed: %v", err)
	}
	first := vecs[0]
	fmt.Printf("First 5 dims: %.4f\n", first[:5])
	fmt.Printf("Dimensions: %d\n\n", len(first))

	// ------------------------------------------------------------------
	// 2. Batch embedding + semantic similarity.
	//    Note similar sentences score high even with no shared words —
	//    that's the whole point of embeddings over keyword search.
	// ------------------------------------------------------------------
	sentences := []string{
		"A dog ran across the yard",
		"A puppy sprinted through the garden",       // similar to [0]
		"The stock market fell sharply today",       // dissimilar
		"Equity markets experienced a steep decline", // similar to [2]
	}

	embs, err := client.Embed(ctx, sentences)
	if err != nil {
		log.Fatalf("batch embed failed: %v", err)
	}

	fmt.Println("Pairwise cosine similarity:")
	for i := 0; i < len(sentences); i++ {
		for j := i + 1; j < len(sentences); j++ {
			sim := CosineSimilarity(embs[i], embs[j])
			fmt.Printf("  [%d]x[%d] %.4f  %q / %q\n", i, j, sim, sentences[i], sentences[j])
		}
	}

	// ------------------------------------------------------------------
	// 3. Cache hit — re-embedding the same text costs nothing.
	// ------------------------------------------------------------------
	start := time.Now()
	if _, err := client.Embed(ctx, sentences); err != nil {
		log.Fatalf("cached embed failed: %v", err)
	}
	fmt.Printf("\nSecond pass (all cache hits) took %v\n", time.Since(start))

	// ------------------------------------------------------------------
	// 4. Matryoshka truncation — trade a little accuracy for smaller,
	//    faster vectors. Compare full-dim vs 256-dim similarity.
	// ------------------------------------------------------------------
	full := CosineSimilarity(embs[0], embs[1])
	small := CosineSimilarity(TruncateDims(embs[0], 256), TruncateDims(embs[1], 256))
	fmt.Printf("\nSimilarity [0]x[1]: full=%.4f  256-dim=%.4f\n", full, small)
}
