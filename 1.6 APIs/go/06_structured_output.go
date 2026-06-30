// 06_structured_output.go — for RAG you rarely want prose; you want data your
// code can use: a parsed entity, a routing decision, a relevance score. Asking
// the model nicely for JSON is unreliable (it adds prose, code fences, apologies).
// response_format: {"type": "json_object"} forces the model to emit ONLY valid
// JSON, which you can unmarshal directly.
//
// Two rules when using JSON mode:
//  1. The prompt MUST contain the word "json" (the API enforces this).
//  2. Describe the exact shape you want in the prompt — the mode guarantees
//     valid JSON, not the schema YOU need.
package main

import (
	"context"
	"encoding/json"
	"fmt"
)

// queryAnalysis is the shape we ask the model to fill in. This is the kind of
// pre-retrieval classification a RAG router does on an incoming question.
type queryAnalysis struct {
	Intent   string   `json:"intent"`    // e.g. "factual", "summarization", "comparison"
	Entities []string `json:"entities"`  // named things to filter/boost on
	NeedsRAG bool     `json:"needs_rag"` // does this require document retrieval at all?
	Rewrite  string   `json:"rewrite"`   // a cleaned-up search query
}

func analyzeQuery(ctx context.Context, c *Client, userQuery string) (*queryAnalysis, error) {
	prompt := fmt.Sprintf(`Analyze the user's search query and respond with a JSON object
containing exactly these fields:
  "intent":    one of "factual", "summarization", "comparison", "chitchat"
  "entities":  array of key named entities in the query
  "needs_rag": boolean, false for greetings/chitchat
  "rewrite":   a concise, keyword-rich version of the query for vector search

User query: %q`, userQuery)

	resp, err := c.Chat(ctx, ChatRequest{
		Model:          "llama-3.1-8b-instant",
		Messages:       []Message{{Role: "user", Content: prompt}},
		ResponseFormat: &ResponseFormat{Type: "json_object"},
		Temperature:    0, // deterministic structured output
		MaxTokens:      512,
	})
	if err != nil {
		return nil, err
	}

	var out queryAnalysis
	raw := resp.Choices[0].Message.Content
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil, fmt.Errorf("model returned non-conforming JSON %q: %w", raw, err)
	}
	return &out, nil
}

func structuredDemo(ctx context.Context, c *Client) error {
	analysis, err := analyzeQuery(ctx, c,
		"how does pgvector's HNSW index compare to IVFFlat for recall?")
	if err != nil {
		return err
	}
	pretty, _ := json.MarshalIndent(analysis, "", "  ")
	fmt.Println(string(pretty))
	return nil
}
