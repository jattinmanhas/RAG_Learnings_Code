import OpenAI from "openai";

const client = new OpenAI({
    baseURL: "http://localhost:11434/v1",
    apiKey: "ollama" // Required by the library but ignored by Ollama
});

const EMBEDDING_MODEL = 'nomic-embed-text:v1.5'; // or 'text-embedding-3-small'


async function main() {
    const response = await client.embeddings.create({
        model: EMBEDDING_MODEL,
        input: "The quick brown fox jumps over the lazy dog.",
        encoding_format: "float"
    });

    console.log(response.data[0].embedding.slice(0, 5)); // first 5 dims
    console.log("Dimensions:", response.data[0].embedding.length);
}

// Batch embedding — always prefer this over looping single calls
// Most providers allow 100-2048 inputs per request
async function batchEmbedding(texts: string[]): Promise<number[][]> {
    const response = await client.embeddings.create({
        model: EMBEDDING_MODEL,
        input: texts,
        encoding_format: 'float',
   });

   return response.data
    .sort((a, b) => a.index - b.index) // ensure order matches input
    .map(item => item.embedding);
  
}

main();

// When vectors are normalized, dot product = cosine similarity
const similarity = (a : number[], b: number[]) => {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    return dot; // if normalized, this is cosine similarity
}

const sentences = [
    'A dog ran across the yard',
    'A puppy sprinted through the garden',   // should be similar to [0]
    'The stock market fell sharply today',    // should be dissimilar
    'Equity markets experienced a steep decline', // should be similar to [2]
];

async function testBatchEmbedding() {
    const embeddings = await batchEmbedding(sentences);

      // Compare all pairs
    for (let i = 0; i < sentences.length; i++) {
        for (let j = i + 1; j < sentences.length; j++) {
            const sim = similarity(embeddings[i], embeddings[j]);
            console.log(`Similarity between "${sentences[i]}" and "${sentences[j]}": ${sim.toFixed(4)}`);
        }
    }
}

testBatchEmbedding();

//The .sort() by index — good instinct adding that. The OpenAI spec doesn't guarantee response order matches input order, so that's a production-correctness thing, not just cleanliness.
//The dot product = cosine similarity comment — this only holds because nomic-embed-text:v1.5 outputs L2-normalized vectors (unit vectors). If you switched to a model that doesn't normalize, you'd get wrong results silently.
//The batch embedding function is a great example of how to efficiently get embeddings for multiple inputs. Always prefer batching over looping single calls for performance and cost reasons.