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
 * Custom Retriever Example
 * 
 * This demonstrates how to customize the retrieval process:
 * 1. Hybrid retrieval (combining multiple strategies)
 * 2. Metadata-based filtering
 * 3. Custom scoring/ranking
 * 4. Query expansion
 */

interface RetrievalOptions {
  k?: number;
  scoreThreshold?: number;
  metadataFilter?: Record<string, any>;
}

class CustomRAGSystem {
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

  async setupVectorStore(documentsPath: string): Promise<void> {
    const documents: Document[] = [];
    const files = fs.readdirSync(documentsPath);

    for (const file of files) {
      if (file.endsWith('.txt')) {
        const filePath = path.join(documentsPath, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        
        documents.push(
          new Document({
            pageContent: content,
            metadata: { 
              source: file,
              category: this.categorize(file),
              length: content.length,
              words: content.split(/\s+/).length
            },
          })
        );
      }
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await textSplitter.splitDocuments(documents);
    this.vectorStore = await HNSWLib.fromDocuments(splitDocs, this.embeddings);
    
    console.log(`✓ Indexed ${splitDocs.length} chunks from ${documents.length} documents`);
  }

  private categorize(filename: string): string {
    if (filename.includes('langchain')) return 'framework';
    if (filename.includes('rag')) return 'concept';
    if (filename.includes('vector')) return 'database';
    return 'general';
  }

  /**
   * Custom Retrieval Strategy 1: Query Expansion
   * Generate multiple variations of the query to improve retrieval
   */
  async retrieveWithQueryExpansion(
    query: string, 
    options: RetrievalOptions = {}
  ): Promise<Document[]> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    const k = options.k || 3;
    
    const expansionPrompt = PromptTemplate.fromTemplate(
      `Given this question, generate 2 alternative ways to ask the same question:

Original: {question}

Alternative 1:
Alternative 2:`
    );

    const expansionChain = RunnableSequence.from([
      expansionPrompt,
      this.llm,
      new StringOutputParser(),
    ]);

    console.log(`\n🔄 Expanding query: "${query}"`);
    const expansions = await expansionChain.invoke({ question: query });
    
    const queries = [query, ...expansions.split('\n').filter(q => q.trim())];
    console.log(`Generated ${queries.length} query variations`);

    const allResults: Document[] = [];
    const seenContent = new Set<string>();

    for (const q of queries) {
      const results = await this.vectorStore.similaritySearch(q, k);
      
      for (const doc of results) {
        if (!seenContent.has(doc.pageContent)) {
          seenContent.add(doc.pageContent);
          allResults.push(doc);
        }
      }
    }

    return allResults.slice(0, k * 2);
  }

  /**
   * Custom Retrieval Strategy 2: Metadata Filtering
   * Filter results by metadata attributes
   */
  async retrieveWithMetadataFilter(
    query: string,
    options: RetrievalOptions = {}
  ): Promise<Document[]> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    const k = options.k || 10;
    const results = await this.vectorStore.similaritySearch(query, k);

    if (!options.metadataFilter) {
      return results;
    }

    console.log(`\n🔍 Filtering by metadata:`, options.metadataFilter);

    const filtered = results.filter(doc => {
      for (const [key, value] of Object.entries(options.metadataFilter!)) {
        if (doc.metadata[key] !== value) {
          return false;
        }
      }
      return true;
    });

    console.log(`Filtered from ${results.length} to ${filtered.length} results`);
    return filtered;
  }

  /**
   * Custom Retrieval Strategy 3: Hybrid Retrieval with Reranking
   * Retrieve more documents, then rerank based on custom criteria
   */
  async retrieveWithReranking(
    query: string,
    options: RetrievalOptions = {}
  ): Promise<Document[]> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    const initialK = (options.k || 3) * 3;
    const finalK = options.k || 3;

    console.log(`\n🎯 Initial retrieval: ${initialK} documents`);
    const results = await this.vectorStore.similaritySearchWithScore(query, initialK);

    const queryLower = query.toLowerCase();
    const queryWords = new Set(queryLower.split(/\s+/));

    const reranked = results.map(([doc, score]) => {
      const content = doc.pageContent.toLowerCase();
      
      let keywordScore = 0;
      for (const word of queryWords) {
        if (content.includes(word)) {
          keywordScore += 1;
        }
      }

      const recencyScore = doc.metadata.category === 'concept' ? 0.1 : 0;
      
      const finalScore = score - (keywordScore * 0.1) - recencyScore;

      return { doc, originalScore: score, finalScore };
    });

    reranked.sort((a, b) => a.finalScore - b.finalScore);

    console.log(`🏆 Top ${finalK} after reranking:`);
    reranked.slice(0, finalK).forEach((item, i) => {
      console.log(`  ${i + 1}. ${item.doc.metadata.source} (score: ${item.finalScore.toFixed(4)})`);
    });

    return reranked.slice(0, finalK).map(item => item.doc);
  }

  /**
   * Custom Retrieval Strategy 4: Ensemble Retrieval
   * Combine results from multiple retrieval methods
   */
  async retrieveEnsemble(
    query: string,
    options: RetrievalOptions = {}
  ): Promise<Document[]> {
    if (!this.vectorStore) throw new Error('Vector store not initialized');

    console.log(`\n🎭 Ensemble retrieval for: "${query}"`);

    const [basicResults, expandedResults] = await Promise.all([
      this.vectorStore.similaritySearch(query, 5),
      this.retrieveWithQueryExpansion(query, { k: 3 })
    ]);

    const contentToDoc = new Map<string, { doc: Document; votes: number }>();

    for (const doc of basicResults) {
      contentToDoc.set(doc.pageContent, { doc, votes: 2 });
    }

    for (const doc of expandedResults) {
      const existing = contentToDoc.get(doc.pageContent);
      if (existing) {
        existing.votes += 1;
      } else {
        contentToDoc.set(doc.pageContent, { doc, votes: 1 });
      }
    }

    const ranked = Array.from(contentToDoc.values())
      .sort((a, b) => b.votes - a.votes);

    console.log(`Combined ${ranked.length} unique documents`);
    ranked.slice(0, 5).forEach((item, i) => {
      console.log(`  ${i + 1}. ${item.doc.metadata.source} (votes: ${item.votes})`);
    });

    return ranked.slice(0, options.k || 3).map(item => item.doc);
  }

  async queryWithCustomRetrieval(
    question: string,
    retrievalStrategy: 'expansion' | 'metadata' | 'reranking' | 'ensemble',
    options: RetrievalOptions = {}
  ): Promise<string> {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Strategy: ${retrievalStrategy.toUpperCase()}`);
    console.log(`Question: ${question}`);
    console.log('='.repeat(60));

    let relevantDocs: Document[];

    switch (retrievalStrategy) {
      case 'expansion':
        relevantDocs = await this.retrieveWithQueryExpansion(question, options);
        break;
      case 'metadata':
        relevantDocs = await this.retrieveWithMetadataFilter(question, options);
        break;
      case 'reranking':
        relevantDocs = await this.retrieveWithReranking(question, options);
        break;
      case 'ensemble':
        relevantDocs = await this.retrieveEnsemble(question, options);
        break;
    }

    const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');

    const promptTemplate = PromptTemplate.fromTemplate(
      `Answer based on this context:
      
Context:
{context}

Question: {question}

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
    console.log(`\n💡 Answer: ${answer}\n`);
    
    return answer;
  }
}

async function main() {
  try {
    console.log('🎨 Custom Retrieval Strategies Demo\n');

    const ragSystem = new CustomRAGSystem();
    const documentsPath = path.join(__dirname, '../documents');

    console.log('Setting up vector store...');
    await ragSystem.setupVectorStore(documentsPath);

    const question = 'How do vector databases work?';

    await ragSystem.queryWithCustomRetrieval(question, 'expansion');
    
    await ragSystem.queryWithCustomRetrieval(question, 'reranking');
    
    await ragSystem.queryWithCustomRetrieval(question, 'ensemble');

    await ragSystem.queryWithCustomRetrieval(
      'Tell me about LangChain',
      'metadata',
      { metadataFilter: { category: 'framework' } }
    );

    console.log('✅ Demo completed!\n');
    console.log('💡 Key Insights:');
    console.log('  • Different retrieval strategies excel in different scenarios');
    console.log('  • Query expansion helps with varied phrasings');
    console.log('  • Reranking improves relevance with custom scoring');
    console.log('  • Ensemble methods combine multiple approaches');
    console.log('  • Metadata filtering narrows results to specific categories\n');

  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

main();
