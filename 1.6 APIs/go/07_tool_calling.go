// 07_tool_calling.go — tool (a.k.a. function) calling is the foundation of
// agentic RAG. You describe functions to the model; instead of answering, it can
// reply "call search_docs with {query: ...}". YOUR code runs the function and
// feeds the result back. The model then writes the final answer grounded in that
// result.
//
// The loop is:
//  1. Send messages + tool definitions.
//  2. If the model returns tool_calls, execute each locally.
//  3. Append the results as role:"tool" messages and call again.
//  4. Repeat until the model answers with plain content.
package main

import (
	"context"
	"encoding/json"
	"fmt"
)

// ---------------------------------------------------------------------------
// TOOL WIRE TYPES (OpenAI-compatible)
// ---------------------------------------------------------------------------

// Tool describes a function the model is allowed to call.
type Tool struct {
	Type     string       `json:"type"` // always "function"
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"` // JSON Schema object
}

// ToolCall is the model's request to invoke a tool.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // JSON-encoded string of args
	} `json:"function"`
}

// ---------------------------------------------------------------------------
// A LOCAL "TOOL" — here, a fake document search standing in for your retriever.
// ---------------------------------------------------------------------------

func searchDocs(query string) string {
	// In a real RAG system this hits your vector store. We fake a hit so the
	// example is self-contained.
	return fmt.Sprintf(
		"Top result for %q: pgvector supports HNSW and IVFFlat indexes. "+
			"HNSW gives higher recall and faster queries at the cost of slower "+
			"build time and more memory; IVFFlat is cheaper to build.", query)
}

var docSearchTool = Tool{
	Type: "function",
	Function: ToolFunction{
		Name:        "search_docs",
		Description: "Search the knowledge base for passages relevant to a query.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query.",
				},
			},
			"required": []string{"query"},
		},
	},
}

func toolCallingDemo(ctx context.Context, c *Client) error {
	messages := []Message{
		{Role: "system", Content: "You answer using the search_docs tool when you need facts. Cite what you find."},
		{Role: "user", Content: "Which pgvector index gives better recall, HNSW or IVFFlat?"},
	}

	// Allow a few rounds so the model can search then answer.
	for round := 0; round < 4; round++ {
		resp, err := c.Chat(ctx, ChatRequest{
			Model:    "llama-3.3-70b-versatile", // tool use wants a stronger model
			Messages: messages,
			Tools:    []Tool{docSearchTool},
		})
		if err != nil {
			return err
		}

		msg := resp.Choices[0].Message

		// No tool calls → the model gave its final answer.
		if len(msg.ToolCalls) == 0 {
			fmt.Println(msg.Content)
			return nil
		}

		// Echo the assistant turn (with its tool_calls) back into history.
		messages = append(messages, msg)

		// Execute each requested tool and append the result.
		for _, call := range msg.ToolCalls {
			var args struct {
				Query string `json:"query"`
			}
			_ = json.Unmarshal([]byte(call.Function.Arguments), &args)

			var result string
			switch call.Function.Name {
			case "search_docs":
				result = searchDocs(args.Query)
			default:
				result = fmt.Sprintf("unknown tool %q", call.Function.Name)
			}

			fmt.Printf("  [called %s(%q)]\n", call.Function.Name, args.Query)
			messages = append(messages, Message{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    result,
			})
		}
	}

	return fmt.Errorf("tool loop did not converge")
}
