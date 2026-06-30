// 04_streaming_timeout.go — streaming prints tokens as they are generated for a
// snappy UX, but introduces a new failure mode: the stream can stall mid-answer
// (upstream hiccup) without ever erroring. We guard against that with a
// "silence timeout": if no token arrives within N seconds, we abort.
//
// We implement it with a context that we reset on every received token. A
// background goroutine cancels the request when the silence deadline passes.
package main

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"
)

func streamWithTimeout(parent context.Context, c *Client, prompt string, silence time.Duration) (string, error) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	// A resettable watchdog. Each token pushes the deadline forward; if the
	// timer ever fires, we cancel ctx, which aborts the HTTP read.
	var mu sync.Mutex
	timer := time.AfterFunc(silence, cancel)
	reset := func() {
		mu.Lock()
		timer.Reset(silence)
		mu.Unlock()
	}

	full, err := c.Stream(ctx, ChatRequest{
		Model:     "llama-3.1-8b-instant",
		Messages:  []Message{{Role: "user", Content: prompt}},
		MaxTokens: 512,
	}, func(token string) {
		reset() // a token arrived — the stream is alive
		fmt.Fprint(os.Stdout, token)
	})

	timer.Stop()
	fmt.Println()

	// Distinguish a real silence stall from normal completion / caller cancel.
	if err != nil && ctx.Err() == context.Canceled && parent.Err() == nil {
		return full, fmt.Errorf("stream went silent for %s — possible upstream stall", silence)
	}
	return full, err
}

func streamDemo(ctx context.Context, c *Client) error {
	_, err := streamWithTimeout(ctx, c, "Tell me a joke", 10*time.Second)
	return err
}
