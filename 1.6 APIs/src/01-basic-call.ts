import { client } from './client.js';

const response = await client.chat.completions.create({
    model: "llama-3.1-8b-instant",
    messages: [
        { role: "system", content: "You are a concise assistant." },
        { role: "user", content: "What is a context window?" },
    ],
    temperature: 0.2,
    max_tokens: 256
});

console.log(response.choices[0].message.content);