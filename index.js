import { Configuration, OpenAIApi } from 'openai';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

async function main() {
  try {
    const response = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: "Your text string goes here",
    });

    const embeddingData = response.data.data[0].embedding;
    console.log("Embedding:", embeddingData);
    console.log("Model:", response.data.model);
    console.log("Usage:", response.data.usage);
    
    console.log(response.data);
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

main();
