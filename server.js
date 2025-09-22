import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import * as dotenv from 'dotenv';
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());

const ai = new GoogleGenAI({});
const History = [];

async function transformQuery(question) {
  History.push({
    role: 'user',
    parts: [{ text: question }]
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
Only output the rewritten question and nothing else.`
    }
  });

  History.pop(); // remove the rephrasing request from history
  return response.text;
}

async function processQuestion(question) {
  const queries = await transformQuery(question);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  const queryVector = await embeddings.embedQuery(queries);

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  const context = searchResults.matches
    .map(match => match.metadata.text)
    .join("\n\n---\n\n");

  History.push({
    role: 'user',
    parts: [{ text: queries }]
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Jio Games QA expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.
If the answer is not in the context, you must say "I could not find the answer in the provided document."
Keep your answers clear, concise, and educational.

Context: ${context}`
    }
  });

  const answer = response.text;

  History.push({
    role: 'model',
    parts: [{ text: answer }]
  });

  return answer;
}

app.post('/ask', async (req, res) => {
  const question = req.body.question;

  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  try {
    const answer = await processQuestion(question);
    res.json({ answer });
  } catch (err) {
    console.error('Error processing question:', err);
    res.status(500).json({ error: 'Failed to process question' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
