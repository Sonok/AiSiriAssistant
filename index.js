import { Configuration, OpenAIApi } from 'openai';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

async function getEmbedding(word) {
  const response = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: word,
  });
  // Log token usage
  console.log(`Token usage for word "${word}":`, response.data.usage);

  return response.data.data[0].embedding;
}

async function main() {
  try {
    const king = await getEmbedding("king");
    const man = await getEmbedding("man");
    const woman = await getEmbedding("woman");

    const result = king.map((value, index) => value - man[index] + woman[index]);

    console.log("Resultant embedding for 'king - man + woman':", result);

    // Assuming we have a vocabulary of words to compare
    const vocabulary = ["queen", "princess", "throne", "duchess", "monarch", "royalty", "lady", "ruler", "female", "empress"];
    const embeddings = await Promise.all(vocabulary.map(getEmbedding));

    // Compute cosine similarity and find top 10 most similar words
    const similarities = embeddings.map((embedding, i) => ({
      word: vocabulary[i],
      similarity: cosineSimilarity(result, embedding),
    }));

    similarities.sort((a, b) => b.similarity - a.similarity);

    console.log("Top 10 most similar words:");
    console.log(similarities.slice(0, 10));
  } catch (error) {
    if (error.response && error.response.status === 429) {
      console.error("Error: You exceeded your current quota. Please check your plan and billing details.");
    } else if (error.response && error.response.status === 401) {
      console.error("Error: Invalid API key provided. Please check your API key.");
    } else {
      console.error("Error creating embedding:", error);
    }
  }
}

function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

main();
