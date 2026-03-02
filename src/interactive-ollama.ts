import * as readline from 'readline';
import * as path from 'path';
import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

/**
 * Interactive RAG System with Ollama (FREE!)
 * Allows you to ask questions interactively via the command line
 */

class InteractiveRAG {
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

  async loadVectorStore(directory: string): Promise<void> {
    try {
      this.vectorStore = await HNSWLib.load(directory, this.embeddings);
      console.log('✓ Vector store loaded successfully');
    } catch (error) {
      console.error('❌ Failed to load vector store:', error);
      console.log('\n💡 Tip: Run "yarn run ollama" first to create the vector store');
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
  console.log('Interactive RAG System)\n');
  
  const ragSystem = new InteractiveRAG();
  const vectorStorePath = path.join(__dirname, '../vector_store_ollama');

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
        console.log('\nGoodbye!');
        rl.close();
        return;
      }

      if (question.toLowerCase() === 'verbose') {
        verbose = !verbose;
        console.log(`\n🔧 Verbose mode: ${verbose ? 'ON' : 'OFF'}\n`);
        askQuestion();
        return;
      }

      if (question.toLowerCase() === 'help') {
        console.log('\nExample Questions You Can Ask:\n');
        console.log('About LangChain:');
        console.log('  • "What is LangChain?"');
        console.log('  • "What features does LangChain have?"\n');
        console.log('About RAG:');
        console.log('  • "What is RAG?"');
        console.log('  • "How does RAG work?"\n');
        console.log('About Vector Databases:');
        console.log('  • "What is a vector database?"\n');
        console.log('About Peaky Blinders:');
        console.log('  • "What is Peaky Blinders about?"');
        console.log('  • "Who is Tommy Shelby?"');
        console.log('  • "Who are the main characters?"\n');
        console.log('💡 Type "verbose" to see which documents are used.\n');
        askQuestion();
        return;
      }

      try {
        console.log('\nFikir jap bro...');
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
