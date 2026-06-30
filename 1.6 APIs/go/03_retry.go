// 03_retry.go — production calls fail transiently: rate limits (429), overloaded
// servers (500/503), timeouts (408). The fix is to retry, but only on
// retryable statuses, with exponential backoff + jitter, and to obey the
// Retry-After header when the API tells us exactly how long to wait.
package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"time"
)

var retryableStatus = map[int]bool{429: true, 500: true, 503: true, 408: true}

func callWithRetry(ctx context.Context, c *Client, prompt string, maxRetries int) (string, error) {
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err := c.Chat(ctx, ChatRequest{
			Model:     "llama-3.1-8b-instant",
			Messages:  []Message{{Role: "user", Content: prompt}},
			MaxTokens: 512,
		})
		if err == nil {
			return resp.Choices[0].Message.Content, nil
		}
		lastErr = err

		// Only retry known-transient API errors. Anything else (bad request,
		// auth failure, network error) is fatal — fail fast.
		var apiErr *APIError
		if !errors.As(err, &apiErr) || !retryableStatus[apiErr.StatusCode] {
			return "", err
		}

		delay := backoff(attempt, apiErr.RetryAfter)
		fmt.Printf("Attempt %d failed (%d). Retrying in %.0fs...\n",
			attempt+1, apiErr.StatusCode, delay.Seconds())

		// Sleep, but wake early if the caller cancels the context.
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}

	return "", fmt.Errorf("exhausted %d retries: %w", maxRetries, lastErr)
}

// backoff decides how long to wait before the next attempt.
func backoff(attempt int, retryAfter string) time.Duration {
	// The API told us exactly how long to wait — obey it.
	if retryAfter != "" {
		if secs, err := strconv.Atoi(retryAfter); err == nil {
			return time.Duration(secs) * time.Second
		}
	}

	// Otherwise: exponential backoff with jitter, capped at 30s.
	const base = time.Second
	const maxDelay = 30 * time.Second
	exponential := time.Duration(math.Pow(2, float64(attempt))) * base
	jitter := time.Duration(rand.Intn(1000)) * time.Millisecond
	if d := exponential + jitter; d < maxDelay {
		return d
	}
	return maxDelay
}

func retryDemo(ctx context.Context, c *Client) error {
	result, err := callWithRetry(ctx, c, "Explain RAG in one sentence.", 3)
	if err != nil {
		return err
	}
	fmt.Println(result)
	return nil
}
