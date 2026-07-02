// cache.go — a tiny content-addressed embedding cache.
//
// Embeddings are deterministic for a given (model, text): the same input always
// produces the same vector. That makes them ideal to cache. In production this
// saves real money on re-ingested documents and repeated user queries — swap
// this map for Redis / a KV store to share the cache across instances.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

type EmbeddingCache struct {
	mu   sync.RWMutex
	data map[string][]float64
}

func NewEmbeddingCache() *EmbeddingCache {
	return &EmbeddingCache{data: make(map[string][]float64)}
}

// key includes the model so switching models doesn't serve stale vectors.
func key(text string) string {
	sum := sha256.Sum256([]byte(embeddingModel + "\x00" + text))
	return hex.EncodeToString(sum[:])
}

func (c *EmbeddingCache) Get(text string) ([]float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.data[key(text)]
	return v, ok
}

func (c *EmbeddingCache) Set(text string, vec []float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key(text)] = vec
}
