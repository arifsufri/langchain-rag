import * as fs from 'fs';
import * as path from 'path';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

/**
 * Streaming Responses with Ollama (FREE!)
 * 
 * This demonstrates how to stream responses token-by-token
 * for a better user experience
 */

class StreamingRAG {
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

    return documents;
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

  async streamQuery(question: string): Promise<void> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    console.log('\n📚 Retrieving relevant documents...');
    const relevantDocs = await this.vectorStore.similaritySearch(question, 3);

    console.log(`✓ Found ${relevantDocs.length} relevant documents\n`);
    relevantDocs.forEach((doc, i) => {
      console.log(`  ${i + 1}. ${doc.metadata.source}`);
    });

    const context = relevantDocs.map((doc) => doc.pageContent).join('\n\n');

    const promptTemplate = PromptTemplate.fromTemplate(
      `You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:`
    );

    const chain = RunnableSequence.from([
      promptTemplate,
      this.llm,
      new StringOutputParser(),
    ]);

    console.log('\n🤖 Assistant: ');
    
    const stream = await chain.stream({
      context,
      question,
    });

    for await (const chunk of stream) {
      process.stdout.write(chunk);
    }

    console.log('\n');
  }
}

async function main() {
  try {
    console.log('🌊 Streaming RAG Demo with Ollama (FREE!)\n');

    const ragSystem = new StreamingRAG();
    const documentsPath = path.join(__dirname, '../documents');

    console.log('Loading documents...');
    const documents = await ragSystem.loadDocuments(documentsPath);
    
    console.log('Creating vector store with Ollama...');
    await ragSystem.createVectorStore(documents);
    console.log('✓ Ready!\n');

    const questions = [
      'What is LangChain and why is it useful?',
      'Explain how RAG systems work',
      'What are the benefits of using vector databases?',
    ];

    for (const question of questions) {
      console.log('='.repeat(60));
      console.log(`\n❓ Question: ${question}`);
      console.log('─'.repeat(60));
      
      await ragSystem.streamQuery(question);
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.log('='.repeat(60));
    console.log('\n✅ Streaming demo completed!');
    console.log('\n💡 Benefits of streaming:');
    console.log('  • Better user experience - see responses as they generate');
    console.log('  • Lower perceived latency');
    console.log('  • Works great for long responses');
    console.log('  • Completely FREE with Ollama!\n');

  } catch (error) {
    console.error('❌ Error:', error);
    console.log('\n💡 Make sure Ollama is running: ollama serve');
    process.exit(1);
  }
}

main();
