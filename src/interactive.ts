import * as readline from 'readline';
import * as path from 'path';
import { config } from 'dotenv';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

config();

/**
 * Interactive RAG System
 * Allows you to ask questions interactively via the command line
 */

class InteractiveRAG {
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

  async loadVectorStore(directory: string): Promise<void> {
    try {
      this.vectorStore = await HNSWLib.load(directory, this.embeddings);
      console.log('✓ Vector store loaded successfully');
    } catch (error) {
      console.error('❌ Failed to load vector store:', error);
      console.log('\n💡 Tip: Run "npm start" first to create the vector store');
      process.exit(1);
    }
  }

  async query(question: string, verbose: boolean = false): Promise<string> {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    const relevantDocs = await this.vectorStore.similaritySearch(question, 3);

    if (verbose) {
      console.log(`\n📚 Found ${relevantDocs.length} relevant documents:`);
      relevantDocs.forEach((doc, i) => {
        console.log(`  ${i + 1}. ${doc.metadata.source}`);
      });
    }

    const context = relevantDocs.map((doc) => doc.pageContent).join('\n\n');

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
    return answer.trim();
  }
}

async function main() {
  console.log('🤖 Interactive RAG System\n');
  
  const ragSystem = new InteractiveRAG();
  const vectorStorePath = path.join(__dirname, '../vector_store');

  console.log('Loading vector store...');
  await ragSystem.loadVectorStore(vectorStorePath);

  console.log('\n' + '='.repeat(60));
  console.log('You can now ask questions! Type "exit" or "quit" to stop.');
  console.log('Type "verbose" to toggle detailed output.');
  console.log('='.repeat(60) + '\n');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let verbose = false;

  const askQuestion = () => {
    rl.question('You: ', async (input) => {
      const question = input.trim();

      if (!question) {
        askQuestion();
        return;
      }

      if (question.toLowerCase() === 'exit' || question.toLowerCase() === 'quit') {
        console.log('\n👋 Goodbye!');
        rl.close();
        return;
      }

      if (question.toLowerCase() === 'verbose') {
        verbose = !verbose;
        console.log(`\n🔧 Verbose mode: ${verbose ? 'ON' : 'OFF'}\n`);
        askQuestion();
        return;
      }

      try {
        const answer = await ragSystem.query(question, verbose);
        console.log(`\nAssistant: ${answer}\n`);
      } catch (error) {
        console.error('❌ Error:', error);
      }

      askQuestion();
    });
  };

  askQuestion();
}

main();
