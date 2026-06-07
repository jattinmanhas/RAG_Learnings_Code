import { client } from "./client.js";

const RETRYABLE_STATUS_CODES = new Set([429, 500, 503, 408]);

async function callWithRetry(
    prompt: string,
    maxRetries: number = 3,
): Promise<string> {
    let lastError: any = new Error("Unknown error");

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const response = await client.chat.completions.create({
                model: "llama-3.1-8b-instant",
                messages: [{ role: "user", content: prompt }],
                max_tokens: 512,
            });

            return response.choices[0].message.content ?? "";
        } catch (error: any) {
            lastError = error;

            const status = error?.status;

            // Don't retry non-retryable errors
            if (status && !RETRYABLE_STATUS_CODES.has(status)) {
                throw error;
            }


            // On rate limit, respect the Retry-After header if present
            const retryAfter = error?.headers?.["retry-after"]; // API TOLD US TO WAIT, LET'S OBEY
            let delay: number;

            if (retryAfter) {
                delay = parseInt(retryAfter, 10) * 1000; // Convert seconds to ms
            } else {
                // Exponential backoff + jitter
                const base = 1000;                           // 1 second base
                const maxDelay = 30_000;                     // 30 seconds ceiling
                const exponential = base * Math.pow(2, attempt);
                const jitter = Math.random() * 1000;         // 0–1000ms random
                delay = Math.min(exponential + jitter, maxDelay);
            }

            console.log(`Attempt ${attempt + 1} failed (${status}). Retrying in ${Math.round(delay / 1000)}s...`);

            await new Promise(resolve => setTimeout(resolve, delay));

        }
    }

    throw lastError;
}


// Usage
try {
    const result = await callWithRetry("Explain RAG in one sentence.");
    console.log(result);
} catch (err: any) {
    console.error("Call failed:", err);

    // If the thrown value is a plain object (some SDKs throw structured objects),
    // print its enumerable and non-enumerable properties for debugging.
    try {
        const allProps = Object.getOwnPropertyNames(err).reduce((acc: any, k) => {
            acc[k] = (err as any)[k];
            return acc;
        }, {});
        console.error("Error details:", JSON.stringify(allProps, null, 2));
    } catch (e) {
        // ignore
    }

    process.exit(1);
}