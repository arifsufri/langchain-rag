import * as fs from 'fs';
import * as path from 'path';
import { config } from 'dotenv';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

config();

/**
 * Simple RAG System using LangChain TypeScript
 * 
 * This application demonstrates the core concepts of a RAG system:
 * 1. Loading documents from files
 * 2. Splitting text into chunks
 * 3. Creating embeddings and storing in a vector database
 * 4. Retrieving relevant context based on queries
 * 5. Generating responses using retrieved context
 */

class SimpleRAGSystem {
  private vectorStore: HNSWLib | null = null;
  private llm: ChatOpenAI;
  private embeddings: OpenAIEmbeddings;

  constructor() {
    this.llm = new ChatOpenAI({
      modelName: 'gpt-3.5-turbo',
      temperature: 0.7,
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
  }

  /**
   * Load documents from a directory
   * Reads all .txt files and creates Document objects
   */
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
            metadata: { source: file },
          })
        );
      }
    }

    console.log(`✓ Loaded ${documents.length} documents`);
    return documents;
  }

  /**
   * Split documents into smaller chunks
   * This is important because:
   * - Embeddings work better with smaller, focused text chunks
   * - LLMs have context length limits
   * - More precise retrieval of relevant information
   */
  async splitDocuments(documents: Document[]): Promise<Document[]> {
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await textSplitter.splitDocuments(documents);
    console.log(`✓ Split into ${splitDocs.length} chunks`);
    return splitDocs;
  }

  /**
   * Create vector store from documents
   * This converts text chunks into embeddings and stores them
   * in an in-memory vector database (HNSWLib)
   */
  async createVectorStore(documents: Document[]): Promise<void> {
    console.log('Creating vector store...');
    this.vectorStore = await HNSWLib.fromDocuments(
      documents,
      this.embeddings
    );
    console.log('✓ Vector store created successfully');
  }

  /**
   * Save vector store to disk
   * This allows you to reuse the embeddings without recreating them
   */
  async saveVectorStore(directory: string): Promise<void> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }
    await this.vectorStore.save(directory);
    console.log(`✓ Vector store saved to ${directory}`);
  }

  /**
   * Load vector store from disk
   */
  async loadVectorStore(directory: string): Promise<void> {
    this.vectorStore = await HNSWLib.load(
      directory,
      this.embeddings
    );
    console.log(`✓ Vector store loaded from ${directory}`);
  }

  /**
   * Retrieve relevant documents based on a query
   * Uses similarity search to find the most relevant chunks
   */
  async retrieveDocuments(query: string, k: number = 3): Promise<Document[]> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    const results = await this.vectorStore.similaritySearch(query, k);
    console.log(`\n📚 Retrieved ${results.length} relevant documents:`);
    results.forEach((doc, i) => {
      console.log(`  ${i + 1}. ${doc.metadata.source} (${doc.pageContent.substring(0, 100)}...)`);
    });

    return results;
  }

  /**
   * Generate an answer using RAG
   * This is the core RAG functionality:
   * 1. Retrieve relevant documents
   * 2. Format them as context
   * 3. Pass to LLM with the query
   */
  async query(question: string): Promise<string> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    console.log(`\n❓ Question: ${question}`);

    const relevantDocs = await this.retrieveDocuments(question);

    const context = relevantDocs
      .map((doc) => doc.pageContent)
      .join('\n\n');

    const promptTemplate = PromptTemplate.fromTemplate(
      `You are a helpful assistant that answers questions based on the provided context.
      
Context:
{context}

Question: {question}

Answer the question based on the context above. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Answer:`
    );

    const chain = RunnableSequence.from([
      {
        context: () => context,
        question: (input: { question: string }) => input.question,
      },
      promptTemplate,
      this.llm,
      new StringOutputParser(),
    ]);

    const answer = await chain.invoke({ question });
    
    console.log(`\n💡 Answer: ${answer}`);
    return answer;
  }
}

/**
 * Main function demonstrating the RAG system
 */
async function main() {
  try {
    console.log('🚀 Starting Simple RAG System\n');

    const ragSystem = new SimpleRAGSystem();

    const documentsPath = path.join(__dirname, '../documents');
    const vectorStorePath = path.join(__dirname, '../vector_store');

    console.log('Step 1: Loading documents...');
    const documents = await ragSystem.loadDocuments(documentsPath);

    console.log('\nStep 2: Splitting documents into chunks...');
    const splitDocs = await ragSystem.splitDocuments(documents);

    console.log('\nStep 3: Creating vector store (this may take a moment)...');
    await ragSystem.createVectorStore(splitDocs);

    console.log('\nStep 4: Saving vector store for future use...');
    await ragSystem.saveVectorStore(vectorStorePath);

    console.log('\n' + '='.repeat(60));
    console.log('RAG System is ready! Asking questions...');
    console.log('='.repeat(60));

    await ragSystem.query('What is LangChain?');
    
    await ragSystem.query('How does RAG work?');
    
    await ragSystem.query('What are some popular vector databases?');
    
    await ragSystem.query('What is the capital of France?');

    console.log('\n✅ Demo completed!');
    
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

main();
