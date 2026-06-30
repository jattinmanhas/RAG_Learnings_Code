// 02_multi_turn.go — the API is stateless: it remembers nothing between calls.
// To hold a conversation you resend the ENTIRE history every turn, appending the
// model's reply so the next request carries the full context.
package main

import (
	"context"
	"fmt"
)

// conversation accumulates the running message history.
type conversation struct {
	client   *Client
	messages []Message
}

func newConversation(c *Client, systemPrompt string) *conversation {
	return &conversation{
		client:   c,
		messages: []Message{{Role: "system", Content: systemPrompt}},
	}
}

// say sends one user turn and returns the assistant's reply, mutating history.
func (conv *conversation) say(ctx context.Context, userMessage string) (string, error) {
	conv.messages = append(conv.messages, Message{Role: "user", Content: userMessage})

	resp, err := conv.client.Chat(ctx, ChatRequest{
		Model:       "llama-3.1-8b-instant",
		Messages:    conv.messages, // send full history every time
		Temperature: 0.2,
		MaxTokens:   512,
	})
	if err != nil {
		return "", err
	}

	reply := resp.Choices[0].Message
	conv.messages = append(conv.messages, reply) // remember it for next turn
	return reply.Content, nil
}

func multiTurn(ctx context.Context, c *Client) error {
	conv := newConversation(c, "You are a concise assistant.")

	a1, err := conv.say(ctx, "What is temperature in LLMs?")
	if err != nil {
		return err
	}
	fmt.Println(a1)

	// The model "remembers" because we resent the history above.
	a2, err := conv.say(ctx, "How does it affect hallucinations?")
	if err != nil {
		return err
	}
	fmt.Println(a2)
	return nil
}
