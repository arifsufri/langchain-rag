import * as path from 'path';
import { config } from 'dotenv';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';
import { Document } from '@langchain/core/documents';

config();

/**
 * Streaming RAG Example
 * 
 * Demonstrates how to stream responses from the LLM token-by-token
 * This provides a better user experience for longer responses
 */

class StreamingRAG {
  private vectorStore: HNSWLib | null = null;
  private chatModel: ChatOpenAI;
  private embeddings: OpenAIEmbeddings;

  constructor() {
    this.chatModel = new ChatOpenAI({
      modelName: 'gpt-3.5-turbo',
      temperature: 0.7,
      streaming: true,
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
  }

  async loadVectorStore(directory: string): Promise<void> {
    try {
      this.vectorStore = await HNSWLib.load(directory, this.embeddings);
      console.log('✓ Vector store loaded successfully\n');
    } catch (error) {
      console.error('❌ Failed to load vector store');
      console.log('💡 Run "npm start" first to create the vector store\n');
      process.exit(1);
    }
  }

  /**
   * Query with streaming - tokens appear as they're generated
   */
  async streamQuery(question: string): Promise<void> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`❓ Question: ${question}`);
    console.log('='.repeat(60) + '\n');

    console.log('🔍 Retrieving relevant documents...');
    const relevantDocs = await this.vectorStore.similaritySearch(question, 3);

    console.log(`📚 Found ${relevantDocs.length} relevant chunks:\n`);
    relevantDocs.forEach((doc, i) => {
      console.log(`  ${i + 1}. ${doc.metadata.source}`);
    });

    const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');

    const promptTemplate = PromptTemplate.fromTemplate(
      `You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question: {question}

Provide a detailed answer. If the context doesn't contain enough information, say so.

Answer:`
    );

    const chain = RunnableSequence.from([
      {
        context: () => context,
        question: (input: { question: string }) => input.question,
      },
      promptTemplate,
      this.chatModel,
      new StringOutputParser(),
    ]);

    console.log('\n💬 Streaming response:\n');
    process.stdout.write('Assistant: ');

    const stream = await chain.stream({ question });

    for await (const chunk of stream) {
      process.stdout.write(chunk);
    }

    console.log('\n');
  }

  /**
   * Batch multiple queries and stream each response
   */
  async streamMultipleQueries(questions: string[]): Promise<void> {
    console.log(`\n🎯 Streaming ${questions.length} queries...\n`);

    for (let i = 0; i < questions.length; i++) {
      console.log(`\n[${ i + 1}/${questions.length}]`);
      await this.streamQuery(questions[i]);
      
      if (i < questions.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }

  /**
   * Stream with callbacks for more control
   */
  async streamWithCallbacks(question: string): Promise<void> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`❓ Question: ${question}`);
    console.log('='.repeat(60) + '\n');

    const relevantDocs = await this.vectorStore.similaritySearch(question, 3);
    const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');

    const promptTemplate = PromptTemplate.fromTemplate(
      `Answer concisely based on this context:

Context: {context}

Question: {question}

Answer:`
    );

    const chain = RunnableSequence.from([
      {
        context: () => context,
        question: (input: { question: string }) => input.question,
      },
      promptTemplate,
      this.chatModel,
      new StringOutputParser(),
    ]);

    let tokenCount = 0;
    let fullResponse = '';

    console.log('💬 Streaming with metrics:\n');
    process.stdout.write('Assistant: ');

    const stream = await chain.stream({ question });

    for await (const chunk of stream) {
      process.stdout.write(chunk);
      tokenCount += chunk.split(/\s+/).length;
      fullResponse += chunk;
    }

    console.log('\n');
    console.log(`\n📊 Metrics:`);
    console.log(`  • Tokens (approx): ${tokenCount}`);
    console.log(`  • Characters: ${fullResponse.length}`);
    console.log(`  • Words: ${fullResponse.split(/\s+/).length}`);
  }
}

async function main() {
  try {
    console.log('🌊 Streaming RAG Demo\n');

    const streamingRAG = new StreamingRAG();
    const vectorStorePath = path.join(__dirname, '../vector_store');

    console.log('Loading vector store...');
    await streamingRAG.loadVectorStore(vectorStorePath);

    await streamingRAG.streamQuery(
      'What is LangChain and what are its key features?'
    );

    await streamingRAG.streamWithCallbacks(
      'How does the RAG process work?'
    );

    await streamingRAG.streamMultipleQueries([
      'What are vector databases?',
      'Why is text splitting important?',
    ]);

    console.log('✅ Streaming demo completed!\n');
    console.log('💡 Benefits of streaming:');
    console.log('  • Immediate feedback - users see responses as they generate');
    console.log('  • Better UX for long responses');
    console.log('  • Lower perceived latency');
    console.log('  • Can cancel generation if needed');
    console.log('  • Real-time token counting and monitoring\n');

  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

main();
