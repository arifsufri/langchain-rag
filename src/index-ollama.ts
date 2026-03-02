import * as fs from 'fs';
import * as path from 'path';
import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

/**
 * Simple RAG System using Ollama (FREE Local AI)
 * 
 * This application demonstrates the core concepts of a RAG system using Ollama:
 * 1. Loading documents from files
 * 2. Splitting text into chunks
 * 3. Creating embeddings and storing in a vector database
 * 4. Retrieving relevant context based on queries
 * 5. Generating responses using retrieved context
 * 
 * Setup:
 * 1. Install Ollama: brew install ollama
 * 2. Start Ollama: ollama serve
 * 3. Pull a model: ollama pull llama3.2
 */

class SimpleRAGSystem {
  private vectorStore: HNSWLib | null = null;
  private llm: ChatOllama;
  private embeddings: OllamaEmbeddings;

  constructor() {
    this.llm = new ChatOllama({
      model: 'llama3.2',
      temperature: 0.7,
      baseUrl: 'http://localhost:11434',
    });

    this.embeddings = new OllamaEmbeddings({
      model: 'llama3.2',
      baseUrl: 'http://localhost:11434',
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
            metadata: { source: file },
          })
        );
      }
    }

    console.log(`✓ Loaded ${documents.length} documents`);
    return documents;
  }

  async splitDocuments(documents: Document[]): Promise<Document[]> {
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await textSplitter.splitDocuments(documents);
    console.log(`✓ Split into ${splitDocs.length} chunks`);
    return splitDocs;
  }

  async createVectorStore(documents: Document[]): Promise<void> {
    console.log('Creating vector store with Ollama embeddings...');
    console.log('(This may take a minute on first run)');
    this.vectorStore = await HNSWLib.fromDocuments(
      documents,
      this.embeddings
    );
    console.log('✓ Vector store created successfully');
  }

  async saveVectorStore(directory: string): Promise<void> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }
    await this.vectorStore.save(directory);
    console.log(`✓ Vector store saved to ${directory}`);
  }

  async loadVectorStore(directory: string): Promise<void> {
    this.vectorStore = await HNSWLib.load(directory, this.embeddings);
    console.log('✓ Vector store loaded from disk');
  }

  async retrieveDocuments(query: string, k: number = 3): Promise<Document[]> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    const results = await this.vectorStore.similaritySearch(query, k);
    
    console.log(`\n📚 Retrieved ${results.length} relevant documents:`);
    results.forEach((doc, i) => {
      console.log(`\n[${i + 1}] Source: ${doc.metadata.source}`);
      console.log(`Content preview: ${doc.pageContent.substring(0, 150)}...`);
    });

    return results;
  }

  async queryWithRAG(question: string): Promise<string> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized. Call createVectorStore() first.');
    }

    const relevantDocs = await this.retrieveDocuments(question);

    const context = relevantDocs
      .map((doc) => doc.pageContent)
      .join('\n\n');

    const prompt = PromptTemplate.fromTemplate(
      `You are a helpful assistant. Use the following context to answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:`
    );

    const chain = RunnableSequence.from([
      prompt,
      this.llm,
      new StringOutputParser(),
    ]);

    console.log('\n🤖 Generating answer with Ollama...\n');
    
    const response = await chain.invoke({
      context,
      question,
    });

    return response;
  }
}

async function main() {
  try {
    console.log('🚀 Starting Simple RAG System with Ollama (FREE!)\n');
    console.log('⚡ Using local AI - No API keys or costs required!\n');

    const ragSystem = new SimpleRAGSystem();

    const documentsPath = path.join(__dirname, '../documents');
    const vectorStorePath = path.join(__dirname, '../vector_store_ollama');

    console.log('Step 1: Loading documents...');
    const documents = await ragSystem.loadDocuments(documentsPath);

    console.log('\nStep 2: Splitting documents into chunks...');
    const splitDocs = await ragSystem.splitDocuments(documents);

    console.log('\nStep 3: Creating vector store (this may take a moment)...');
    await ragSystem.createVectorStore(splitDocs);

    await ragSystem.saveVectorStore(vectorStorePath);

    console.log('\n' + '='.repeat(60));
    console.log('Starting RAG Queries');
    console.log('='.repeat(60));

    const questions = [
      'What is LangChain?',
      'How does RAG work?',
      'What are vector databases used for?',
    ];

    for (const question of questions) {
      console.log(`\n${'─'.repeat(60)}`);
      console.log(`\n❓ Question: ${question}`);
      console.log(`${'─'.repeat(60)}`);
      
      const answer = await ragSystem.queryWithRAG(question);
      
      console.log('💡 Answer:');
      console.log(answer);
    }

    console.log('\n' + '='.repeat(60));
    console.log('✅ RAG Demo Completed!');
    console.log('='.repeat(60));
    console.log('\n💡 Tips:');
    console.log('  • Ollama runs completely free on your computer');
    console.log('  • No API keys or internet required after model download');
    console.log('  • Try other models: ollama pull mistral, ollama pull codellama');
    console.log('  • Vector store saved - next run will be faster!');
    console.log('\n📚 Next steps:');
    console.log('  • Run: yarn run ollama-interactive');
    console.log('  • Modify documents/ folder with your own files');
    console.log('  • Try different Ollama models\n');

  } catch (error) {
    console.error('❌ Error:', error);
    console.log('\n💡 Troubleshooting:');
    console.log('  1. Make sure Ollama is running: ollama serve');
    console.log('  2. Check if model is downloaded: ollama list');
    console.log('  3. Download llama3.2: ollama pull llama3.2');
    console.log('  4. Visit: https://ollama.com for more info\n');
    process.exit(1);
  }
}

main();
