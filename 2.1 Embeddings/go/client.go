// client.go — a minimal, dependency-free embeddings client that talks to any
// OpenAI-compatible /v1/embeddings endpoint (Ollama, OpenAI, vLLM, TEI, ...).
//
// Using only net/http keeps the example portable and shows exactly what goes
// over the wire — no SDK magic. In a real service you may prefer the official
// SDK, but the production concerns below (batching, retries, caching,
// normalization) are identical regardless of transport.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"time"
)

const (
	// Point baseURL at Ollama's OpenAI-compatible server. Swap for
	// "https://api.openai.com/v1" + a real key to use OpenAI.
	baseURL        = "http://localhost:11434/v1"
	apiKey         = "ollama" // required by the spec, ignored by Ollama
	embeddingModel = "nomic-embed-text:v1.5"

	// Most providers cap the number of inputs per request (OpenAI ~2048,
	// many self-hosted servers far less). Chunk large workloads to stay
	// under the limit and to bound request size / memory.
	maxBatchSize = 256
)

// Client wraps an HTTP client plus config. Reuse a single Client across the
// process: it shares connection pooling and (optionally) an embedding cache.
type Client struct {
	http  *http.Client
	cache *EmbeddingCache
}

func NewClient() *Client {
	return &Client{
		// Always set an explicit timeout — the zero-value http.Client has
		// none and will hang forever if the server stalls.
		http:  &http.Client{Timeout: 30 * time.Second},
		cache: NewEmbeddingCache(),
	}
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

type embeddingRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
}

type embeddingResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// ---------------------------------------------------------------------------
// Core request with retry + backoff
// ---------------------------------------------------------------------------

// embedOnce sends a single request for a slice of inputs already known to be
// within maxBatchSize. Callers should use Embed, which handles chunking.
func (c *Client) embedOnce(ctx context.Context, inputs []string) ([][]float64, error) {
	body, err := json.Marshal(embeddingRequest{
		Model:          embeddingModel,
		Input:          inputs,
		EncodingFormat: "float",
	})
	if err != nil {
		return nil, err
	}

	// Retry transient failures (429 rate limits, 5xx, network errors) with
	// exponential backoff. Never retry 4xx other than 429 — those are bugs
	// in the request and will fail identically every time.
	const maxRetries = 5
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt-1))) * 200 * time.Millisecond
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/embeddings", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err // network error — retry
			continue
		}
		payload, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("retryable status %d: %s", resp.StatusCode, payload)
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("status %d: %s", resp.StatusCode, payload) // non-retryable
		}

		var parsed embeddingResponse
		if err := json.Unmarshal(payload, &parsed); err != nil {
			return nil, err
		}

		// Sort by index: the OpenAI spec does NOT guarantee response order
		// matches input order. Skipping this silently misaligns vectors
		// with their source text — a nasty, hard-to-spot production bug.
		sort.Slice(parsed.Data, func(i, j int) bool {
			return parsed.Data[i].Index < parsed.Data[j].Index
		})

		out := make([][]float64, len(parsed.Data))
		for i, d := range parsed.Data {
			out[i] = d.Embedding
		}
		return out, nil
	}
	return nil, fmt.Errorf("exhausted retries: %w", lastErr)
}

// Embed returns one vector per input. It transparently chunks large slices to
// respect maxBatchSize and serves previously-seen texts from the cache.
//
// Batching is the single most important cost/latency optimization: one request
// for 256 texts is far cheaper and faster than 256 requests for one text each.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	results := make([][]float64, len(texts))

	// Gather cache misses so we only pay for what we haven't seen before.
	var missIdx []int
	var missText []string
	for i, t := range texts {
		if v, ok := c.cache.Get(t); ok {
			results[i] = v
			continue
		}
		missIdx = append(missIdx, i)
		missText = append(missText, t)
	}

	for start := 0; start < len(missText); start += maxBatchSize {
		end := start + maxBatchSize
		if end > len(missText) {
			end = len(missText)
		}
		vecs, err := c.embedOnce(ctx, missText[start:end])
		if err != nil {
			return nil, err
		}
		for j, v := range vecs {
			idx := missIdx[start+j]
			results[idx] = v
			c.cache.Set(missText[start+j], v)
		}
	}
	return results, nil
}
