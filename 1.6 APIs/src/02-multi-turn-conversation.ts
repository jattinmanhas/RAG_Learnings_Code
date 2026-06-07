import OpenAI from "openai";
import { client } from "./client.js";

const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a concise assistant." }
];

async function chat(userMessage: string) {
    messages.push({ role: "user", content: userMessage });
    
    const response = await client.chat.completions.create({
        model: "llama-3.1-8b-instant",
        messages, // send full history every time
        temperature: 0.2,
        max_tokens: 512
    });

    const assistantMessage = response.choices[0].message;

    // Add assistant reply to history for next turn
    messages.push(assistantMessage);

    return assistantMessage.content;
}

console.log(await chat("What is temperature in LLMs?"));
console.log(await chat("How does it affect hallucinations?"));  // model remembers context