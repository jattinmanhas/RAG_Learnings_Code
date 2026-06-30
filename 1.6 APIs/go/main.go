// main.go — runnable demos for every example in this folder.
//
//	go run .            # lists the available demos
//	go run . basic      # 01 — a single chat completion
//	go run . multiturn  # 02 — holding a conversation
//	go run . retry      # 03 — retry with backoff + Retry-After
//	go run . stream     # 04 — streaming with a silence timeout
//	go run . cost       # 05 — token usage + cost tracking
//	go run . structured # 06 — forced JSON output
//	go run . tools      # 07 — tool / function calling
//
// Needs a real key: put GROQ_API_KEY in 1.6 APIs/.env (free at console.groq.com).
package main

import (
	"context"
	"fmt"
	"os"
	"time"
)

func main() {
	demos := map[string]func(context.Context, *Client) error{
		"basic":      basicCall,
		"multiturn":  multiTurn,
		"retry":      retryDemo,
		"stream":     streamDemo,
		"cost":       costDemo,
		"structured": structuredDemo,
		"tools":      toolCallingDemo,
	}

	if len(os.Args) < 2 {
		fmt.Println("usage: go run . <demo>")
		fmt.Println("demos: basic multiturn retry stream cost structured tools")
		return
	}

	name := os.Args[1]
	demo, ok := demos[name]
	if !ok {
		fmt.Printf("unknown demo %q\n", name)
		os.Exit(1)
	}

	client, err := NewClient()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	// A whole-demo deadline so nothing hangs forever.
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	if err := demo(ctx, client); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
