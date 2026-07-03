// embed.go — a tiny, dependency-free client for Ollama's native embeddings API.
//
// This mirrors getEmbedding() in the TypeScript example: same model, same 768
// dimensions. Using only net/http keeps the focus on vector search rather than
// SDK plumbing. In a real service you'd add batching/retries/caching (see the
// 2.1 Embeddings Go example), but here every embedding call is a query or an
// ingest, so simplicity wins.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// The embedding model. nomic-embed-text:v1.5 returns L2-normalized 768-dim
// vectors, which is why the schema declares vector(768). If you swap models you
// MUST update the column dimension and re-index — pgvector rejects vectors whose
// length doesn't match the column.
const embeddingModel = "nomic-embed-text:v1.5"

var httpClient = &http.Client{Timeout: 30 * time.Second}

func ollamaURL() string {
	if u := os.Getenv("OLLAMA_URL"); u != "" {
		return u
	}
	return "http://localhost:11434"
}

// getEmbedding returns the vector for a single piece of text.
func getEmbedding(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(map[string]string{
		"model":  embeddingModel,
		"prompt": text,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaURL()+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	payload, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama status %d: %s", resp.StatusCode, payload)
	}

	var parsed struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return nil, err
	}
	if len(parsed.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding for %q", text)
	}
	return parsed.Embedding, nil
}
