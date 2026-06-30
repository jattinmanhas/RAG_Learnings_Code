// 05_cost_tracking.go — every response carries a usage block (prompt + completion
// tokens). Multiply by the model's per-token price to know what each call costs,
// and accumulate across a session so a runaway loop doesn't surprise you on the
// bill.
package main

import (
	"context"
	"fmt"
	"time"
)

// price is USD per 1M tokens. Approximate — check current provider pricing.
type price struct{ input, output float64 }

var pricing = map[string]price{
	"gpt-4o":               {input: 2.50, output: 10.00},
	"gpt-4o-mini":          {input: 0.15, output: 0.60},
	"claude-sonnet-4":      {input: 3.00, output: 15.00},
	"llama-3.1-8b-instant": {input: 0.05, output: 0.08}, // Groq pricing
}

// requestCost is the per-call accounting record.
type requestCost struct {
	model        string
	inputTokens  int
	outputTokens int
	costUSD      float64
	duration     time.Duration
}

func trackedCall(ctx context.Context, c *Client, prompt, model string) (string, requestCost, error) {
	if model == "" {
		model = "llama-3.1-8b-instant"
	}

	start := time.Now()
	resp, err := c.Chat(ctx, ChatRequest{
		Model:     model,
		Messages:  []Message{{Role: "user", Content: prompt}},
		MaxTokens: 512,
	})
	if err != nil {
		return "", requestCost{}, err
	}
	elapsed := time.Since(start)

	p := pricing[model] // zero value {0,0} if unknown — cost reads as $0
	usage := resp.Usage
	cost := requestCost{
		model:        model,
		inputTokens:  usage.PromptTokens,
		outputTokens: usage.CompletionTokens,
		costUSD: float64(usage.PromptTokens)/1_000_000*p.input +
			float64(usage.CompletionTokens)/1_000_000*p.output,
		duration: elapsed,
	}

	fmt.Printf("\n    Model:   %s\n    Tokens:  %d in / %d out\n    Cost:    $%.6f\n    Time:    %dms\n",
		model, cost.inputTokens, cost.outputTokens, cost.costUSD, elapsed.Milliseconds())

	return resp.Choices[0].Message.Content, cost, nil
}

func costDemo(ctx context.Context, c *Client) error {
	var sessionCost float64

	_, c1, err := trackedCall(ctx, c, "What is RAG?", "")
	if err != nil {
		return err
	}
	sessionCost += c1.costUSD

	_, c2, err := trackedCall(ctx, c, "What is pgvector?", "")
	if err != nil {
		return err
	}
	sessionCost += c2.costUSD

	fmt.Printf("Total session cost: $%.6f\n", sessionCost)
	return nil
}
