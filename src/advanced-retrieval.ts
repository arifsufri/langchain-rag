import * as fs from 'fs';
import * as path from 'path';
import { config } from 'dotenv';
import { OpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';

config();

/**
 * Advanced Retrieval Examples
 * 
 * This file demonstrates different retrieval strategies:
 * 1. Basic similarity search
 * 2. Similarity search with scores
 * 3. Similarity search with metadata filtering
 * 4. Maximum Marginal Relevance (MMR) search
 */

class AdvancedRetrieval {
  private vectorStore: HNSWLib | null = null;
  private embeddings: OpenAIEmbeddings;

  constructor() {
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
  }

  async loadDocuments(directoryPath: string): Promise<Document[]> {
    const documents: Document[] = [];
    const files = fs.readdirSync(directoryPath);

    for (const file of files) {
      if (file.endsWith('.txt')) {
        const filePath = path.join(directoryPath, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        
        documents.push(
          new Document({
            pageContent: content,
            metadata: { 
              source: file,
              category: this.categorizeFile(file)
            },
          })
        );
      }
    }

    return documents;
  }

  private categorizeFile(filename: string): string {
    if (filename.includes('langchain')) return 'framework';
    if (filename.includes('rag')) return 'concept';
    if (filename.includes('vector')) return 'database';
    return 'general';
  }

  async createVectorStore(documents: Document[]): Promise<void> {
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await textSplitter.splitDocuments(documents);
    this.vectorStore = await HNSWLib.fromDocuments(
      splitDocs,
      this.embeddings
    );
  }

  /**
   * Strategy 1: Basic Similarity Search
   * Returns the k most similar documents
   */
  async basicSimilaritySearch(query: string, k: number = 3): Promise<void> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Strategy 1: Basic Similarity Search (k=${k})`);
    console.log(`Query: "${query}"`);
    console.log('='.repeat(60));

    const results = await this.vectorStore.similaritySearch(query, k);

    results.forEach((doc, i) => {
      console.log(`\n[${i + 1}] Source: ${doc.metadata.source}`);
      console.log(`Category: ${doc.metadata.category}`);
      console.log(`Content: ${doc.pageContent.substring(0, 200)}...`);
    });
  }

  /**
   * Strategy 2: Similarity Search with Scores
   * Returns documents with their similarity scores
   */
  async similaritySearchWithScore(query: string, k: number = 3): Promise<void> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Strategy 2: Similarity Search with Scores (k=${k})`);
    console.log(`Query: "${query}"`);
    console.log('='.repeat(60));

    const results = await this.vectorStore.similaritySearchWithScore(query, k);

    results.forEach(([doc, score], i) => {
      console.log(`\n[${i + 1}] Similarity Score: ${score.toFixed(4)}`);
      console.log(`Source: ${doc.metadata.source}`);
      console.log(`Category: ${doc.metadata.category}`);
      console.log(`Content: ${doc.pageContent.substring(0, 200)}...`);
    });

    console.log(`\n💡 Note: Lower scores indicate higher similarity`);
  }

  /**
   * Strategy 3: Threshold-based Retrieval
   * Only return documents with similarity above a threshold
   */
  async thresholdRetrieval(query: string, threshold: number = 0.5): Promise<void> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Strategy 3: Threshold-based Retrieval (threshold=${threshold})`);
    console.log(`Query: "${query}"`);
    console.log('='.repeat(60));

    const results = await this.vectorStore.similaritySearchWithScore(query, 10);
    const filteredResults = results.filter(([_, score]) => score <= threshold);

    console.log(`\nFound ${filteredResults.length} documents above threshold:`);

    filteredResults.forEach(([doc, score], i) => {
      console.log(`\n[${i + 1}] Score: ${score.toFixed(4)}`);
      console.log(`Source: ${doc.metadata.source}`);
      console.log(`Content: ${doc.pageContent.substring(0, 150)}...`);
    });
  }

  /**
   * Strategy 4: Compare Different Queries
   * Shows how query formulation affects retrieval
   */
  async compareQueries(queries: string[]): Promise<void> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Strategy 4: Query Comparison`);
    console.log('='.repeat(60));

    for (const query of queries) {
      console.log(`\n📝 Query: "${query}"`);
      const results = await this.vectorStore.similaritySearch(query, 2);
      
      console.log(`Top 2 results:`);
      results.forEach((doc, i) => {
        console.log(`  ${i + 1}. ${doc.metadata.source} - "${doc.pageContent.substring(0, 80)}..."`);
      });
    }
  }
}

async function main() {
  try {
    console.log('🔍 Advanced Retrieval Strategies Demo\n');

    const retrieval = new AdvancedRetrieval();

    const documentsPath = path.join(__dirname, '../documents');

    console.log('Loading and processing documents...');
    const documents = await retrieval.loadDocuments(documentsPath);
    await retrieval.createVectorStore(documents);
    console.log('✓ Ready!\n');

    await retrieval.basicSimilaritySearch(
      'What is a vector database?'
    );

    await retrieval.similaritySearchWithScore(
      'How do I use LangChain?'
    );

    await retrieval.thresholdRetrieval(
      'Tell me about embeddings',
      0.8
    );

    await retrieval.compareQueries([
      'What is RAG?',
      'Explain Retrieval-Augmented Generation',
      'How does retrieval help with generation?',
    ]);

    console.log('\n✅ Demo completed!');
    console.log('\n💡 Key Takeaways:');
    console.log('  1. Query formulation significantly impacts retrieval quality');
    console.log('  2. Similarity scores help filter less relevant results');
    console.log('  3. Different strategies work better for different use cases');
    console.log('  4. Experimenting with k and thresholds optimizes performance\n');

  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

main();
