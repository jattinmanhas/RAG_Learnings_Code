// 01_basic_call.go — the smallest useful request: one system prompt, one user
// turn, one answer.
package main

import (
	"context"
	"fmt"
)

func basicCall(ctx context.Context, c *Client) error {
	resp, err := c.Chat(ctx, ChatRequest{
		Model: "llama-3.1-8b-instant",
		Messages: []Message{
			{Role: "system", Content: "You are a concise assistant."},
			{Role: "user", Content: "What is a context window?"},
		},
		Temperature: 0.2,
		MaxTokens:   256,
	})
	if err != nil {
		return err
	}

	fmt.Println(resp.Choices[0].Message.Content)
	return nil
}
