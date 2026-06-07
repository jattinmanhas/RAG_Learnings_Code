import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

export const client = new OpenAI({
    apiKey: process.env.GROQ_API_KEY,
    baseURL: "https://api.groq.com/openai/v1"
});

// Available Groq models (free):
// "llama-3.1-8b-instant"   ← fast, good for learning
// "llama-3.3-70b-versatile" ← smarter, slower
// "mixtral-8x7b-32768"     ← large context window
