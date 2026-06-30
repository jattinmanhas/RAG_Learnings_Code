// client.go — a tiny OpenAI-compatible chat client built on the standard
// library only (net/http + encoding/json). No SDK required.
//
// We point it at Groq, which speaks the same wire protocol as OpenAI but is
// free for learning. Swap baseURL + the env var to target OpenAI, Together,
// Ollama, or any other OpenAI-compatible endpoint.
//
// Available Groq models (free):
//
//	llama-3.1-8b-instant    ← fast, good for learning
//	llama-3.3-70b-versatile ← smarter, slower
//	mixtral-8x7b-32768      ← large context window
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	baseURL      = "https://api.groq.com/openai/v1"
	defaultModel = "llama-3.1-8b-instant"
)

// ---------------------------------------------------------------------------
// WIRE TYPES
// ---------------------------------------------------------------------------
// These mirror the OpenAI Chat Completions schema. We only model the fields we
// actually use; the JSON decoder ignores everything else.

// Message is one turn in a conversation.
type Message struct {
	Role       string     `json:"role"`           // "system" | "user" | "assistant" | "tool"
	Content    string     `json:"content"`        // the text
	Name       string     `json:"name,omitempty"` // tool name (for role "tool")
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ChatRequest is the body POSTed to /chat/completions.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
}

// ResponseFormat forces structured output. Use {"type": "json_object"}.
type ResponseFormat struct {
	Type string `json:"type"`
}

// Usage reports token accounting returned by the API.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatResponse is the (non-streaming) response body.
type ChatResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// APIError is the structured error payload the API returns with 4xx/5xx.
type APIError struct {
	StatusCode int
	RetryAfter string // value of the Retry-After header, if any
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("api error (status %d): %s", e.StatusCode, e.Message)
}

// ---------------------------------------------------------------------------
// CLIENT
// ---------------------------------------------------------------------------

// Client holds the API key and an *http.Client. Reuse one across calls so the
// underlying TCP connections get pooled.
type Client struct {
	apiKey string
	http   *http.Client
}

// NewClient reads GROQ_API_KEY from the environment (loading ../.env first, to
// match the Node example's dotenv behaviour) and returns a ready client.
func NewClient() (*Client, error) {
	loadDotEnv("../.env")

	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("GROQ_API_KEY is not set (put it in 1.6 APIs/.env)")
	}

	return &Client{
		apiKey: key,
		// A generous client timeout guards against a totally dead connection.
		// Per-request deadlines are set via context.Context below.
		http: &http.Client{Timeout: 60 * time.Second},
	}, nil
}

// Chat performs a single non-streaming chat completion.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	if req.Model == "" {
		req.Model = defaultModel
	}

	httpResp, err := c.do(ctx, req)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	body, _ := io.ReadAll(httpResp.Body)

	if httpResp.StatusCode >= 400 {
		return nil, &APIError{
			StatusCode: httpResp.StatusCode,
			RetryAfter: httpResp.Header.Get("Retry-After"),
			Message:    strings.TrimSpace(string(body)),
		}
	}

	var resp ChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	return &resp, nil
}

// Stream performs a streaming chat completion, invoking onToken for every text
// delta as it arrives. It returns the fully concatenated text.
func (c *Client) Stream(ctx context.Context, req ChatRequest, onToken func(string)) (string, error) {
	req.Stream = true
	if req.Model == "" {
		req.Model = defaultModel
	}

	httpResp, err := c.do(ctx, req)
	if err != nil {
		return "", err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= 400 {
		body, _ := io.ReadAll(httpResp.Body)
		return "", &APIError{
			StatusCode: httpResp.StatusCode,
			RetryAfter: httpResp.Header.Get("Retry-After"),
			Message:    strings.TrimSpace(string(body)),
		}
	}

	// The server streams Server-Sent Events: lines of "data: {json}\n",
	// terminated by "data: [DONE]".
	var full strings.Builder
	scanner := bufio.NewScanner(httpResp.Body)
	// Tokens can exceed the default 64KB line buffer on long responses.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue // skip malformed keep-alive lines
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		token := chunk.Choices[0].Delta.Content
		if token != "" {
			full.WriteString(token)
			onToken(token)
		}
	}

	if err := scanner.Err(); err != nil {
		return full.String(), fmt.Errorf("reading stream: %w", err)
	}
	return full.String(), nil
}

// do builds and sends the HTTP request. Cancelling ctx aborts in flight.
func (c *Client) do(ctx context.Context, req ChatRequest) (*http.Response, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	return c.http.Do(httpReq)
}

// loadDotEnv is a minimal .env loader: KEY=VALUE per line, # comments ignored.
// It only sets variables that are not already present in the environment, so
// real env vars always win. Missing file is not an error.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, val)
		}
	}
}
