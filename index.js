import * as dotenv from "dotenv";
dotenv.config();

import fs from "fs";
import path from "path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";

async function indexDocuments() {
  try {
    // Manual PDFs outside the folder
    const manualPDFs = [
     "./JioGames_Ad_SDK_FAQ (1).pdf"
    ];

    // Folder with PDFs
    const PDF_FOLDER = "./pdf_docs";

    let rawDocs = [];

    // Load manual PDFs
    for (const filePath of manualPDFs) {
      if (filePath.toLowerCase().endsWith(".pdf")) {
        const loader = new PDFLoader(filePath);
        const docs = await loader.load();
        rawDocs = rawDocs.concat(docs);
      }
    }

    // Load PDFs from folder
    if (fs.existsSync(PDF_FOLDER)) {
      const folderFiles = fs.readdirSync(PDF_FOLDER);
      for (const file of folderFiles) {
        if (file.toLowerCase().endsWith(".pdf")) {
          const loader = new PDFLoader(path.join(PDF_FOLDER, file));
          const docs = await loader.load();
          rawDocs = rawDocs.concat(docs);
        }
      }
    }

    console.log(`✅ Loaded ${rawDocs.length} documents from manual list + ${PDF_FOLDER}`);

    // Split into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log(`✅ Chunking done: ${chunkedDocs.length} chunks created`);

    // Google Generative AI Embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: "text-embedding-004",
    });
    console.log("✅ Embeddings model initialized");

    // Test embedding dimension
    const testVector = await embeddings.embedQuery("test");
    console.log(`🔹 Embedding dimension: ${testVector.length}`);

    // Validate Pinecone index dimension
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const indexInfo = await pinecone.describeIndex(process.env.PINECONE_INDEX_NAME);
    const expectedDimension = indexInfo.dimension;
    console.log(`🔹 Pinecone index dimension: ${expectedDimension}`);

    if (testVector.length !== expectedDimension) {
      throw new Error(
        `❌ Embedding dimension ${testVector.length} does not match Pinecone index dimension ${expectedDimension}.
         ➜ Solution: Recreate the Pinecone index with dimension ${testVector.length} OR use an embedding model that outputs ${expectedDimension} dimensions.`
      );
    }

    // Store documents in Pinecone
    console.log("⏳ Uploading documents to Pinecone...");
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });
    console.log("✅ Data successfully stored in Pinecone");
  } catch (error) {
    console.error("❌ Error during indexing:", error);
  }
}

indexDocuments();
