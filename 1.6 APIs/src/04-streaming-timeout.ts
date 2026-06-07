import { client } from "./client.js";

async function streamWithTimeout(prompt: string, silenceTimeout = 10_000) {
    const stream = await client.chat.completions.create({
        model: "llama-3.1-8b-instant",
        stream: true,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 512,
    });


    let silenceTimer: NodeJS.Timeout | null = null;
    let fullText = "";

    const resetTimer = () => {
        if (silenceTimer) clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => {
            throw new Error("Stream went silent for 10s — possible upstream stall");
        }, silenceTimeout);
    };

    resetTimer();                      // start timer before first token

    for await (const chunk of stream) {
        resetTimer();                    // reset on every received chunk
        const token = chunk.choices[0]?.delta?.content ?? "";
        fullText += token;
        process.stdout.write(token);
    }

    if (silenceTimer) clearTimeout(silenceTimer);        // clean up when stream ends normally
    return fullText;

}

streamWithTimeout("Tell me a joke").catch(err => {
    console.error("Error during streaming:", err);
});