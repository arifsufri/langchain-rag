# Simple RAG System with LangChain TypeScript

A beginner-friendly implementation of a Retrieval-Augmented Generation (RAG) system using LangChain in TypeScript. This project demonstrates the core concepts of RAG and helps you understand how LangChain works.

> **📚 New here?** Start with [INDEX.md](./INDEX.md) for a complete guide to all documentation.
> 
> **⚡ Quick start?** Jump to [GETTING_STARTED.md](./GETTING_STARTED.md) (5 minutes).

## 🎯 What You'll Learn

- How to load and process documents in LangChain
- Text splitting strategies for optimal retrieval
- Creating and using vector embeddings
- Building a vector store for similarity search
- Implementing the RAG pattern (Retrieve → Augment → Generate)
- Using LangChain's chains and runnables

## 🏗️ Project Structure

```
langchain-rag-ts/
├── src/
│   ├── index.ts                  # Basic RAG implementation (start here!)
│   ├── interactive.ts            # Interactive CLI for asking questions
│   ├── advanced-retrieval.ts     # Different retrieval strategies
│   ├── custom-retrieval.ts       # Advanced: query expansion, reranking, etc.
│   └── streaming.ts              # Streaming responses demo
├── documents/                    # Sample documents for indexing
│   ├── langchain.txt
│   ├── rag.txt
│   └── vector-databases.txt
├── vector_store/                 # Generated vector embeddings (created on first run)
├── package.json
├── tsconfig.json
├── .env                          # API keys configuration
├── README.md                     # This file
└── CONCEPTS.md                   # In-depth LangChain concepts guide
```

## 📋 Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- **Option A:** OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- **Option B:** Ollama for FREE local AI ([Setup guide](./OLLAMA.md)) ⭐ **Recommended for beginners!**

## 📖 Documentation

- **[GETTING_STARTED.md](./GETTING_STARTED.md)** - 5-minute quick start guide
- **[CONCEPTS.md](./CONCEPTS.md)** - Deep dive into LangChain concepts  
- **[PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)** - Complete project overview
- **README.md** (this file) - Technical reference

## 🚀 Getting Started

### Option A: Using Ollama (FREE - No API Key Required) ⭐

**Perfect for learning and testing!**

1. **Install Ollama:**
   ```bash
   brew install ollama
   ```

2. **Start Ollama and download a model:**
   ```bash
   # In terminal 1:
   ollama serve
   
   # In terminal 2:
   ollama pull llama3.2
   ```

3. **Run the project:**
   ```bash
   yarn install
   yarn run ollama
   ```

**📖 All Ollama Commands:** See [OLLAMA-COMMANDS.md](./OLLAMA-COMMANDS.md) for interactive mode, advanced retrieval, and more!

**Available Ollama commands:**
- `yarn run ollama` - Basic RAG demo
- `yarn run ollama:interactive` - Interactive Q&A ⭐
- `yarn run ollama:advanced` - Advanced retrieval strategies
- `yarn run ollama:streaming` - Streaming responses

**📖 Full Ollama setup guide:** [OLLAMA.md](./OLLAMA.md)

---

### Option B: Using OpenAI (Requires API Key)

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment Variables

Edit the `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys).

### 3. Run the Application

The project includes multiple examples to help you learn different aspects of RAG:

#### Basic RAG Demo
```bash
# With OpenAI:
npm start

# With Ollama (FREE):
npm run ollama
```
This creates a vector store from sample documents and demonstrates basic RAG queries.

#### Interactive Mode
```bash
npm run interactive
```
Ask questions interactively in your terminal. Great for experimentation!

#### Advanced Retrieval
```bash
npm run advanced
```
Explore different retrieval strategies: similarity search, scoring, thresholds, and query comparison.

#### Custom Retrieval
```bash
npm run custom
```
See advanced techniques: query expansion, metadata filtering, reranking, and ensemble retrieval.

#### Streaming Responses
```bash
npm run streaming
```
Learn how to stream LLM responses token-by-token for better UX.

## 🔍 How It Works

### Step 1: Document Loading
The system reads text files from the `documents/` directory and converts them into LangChain `Document` objects.

```typescript
const documents = await ragSystem.loadDocuments(documentsPath);
```

### Step 2: Text Splitting
Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`. This improves retrieval accuracy and fits within LLM context limits.

```typescript
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,      // Characters per chunk
  chunkOverlap: 50,    // Overlap to maintain context
});
```

### Step 3: Creating Embeddings
Text chunks are converted to vector embeddings using OpenAI's embedding model and stored in an in-memory vector database (HNSWLib).

```typescript
this.vectorStore = await HNSWLib.fromDocuments(
  documents,
  this.embeddings
);
```

### Step 4: Querying with RAG
When you ask a question:
1. **Retrieve**: The system finds the most similar documents using semantic search
2. **Augment**: Retrieved documents are formatted as context
3. **Generate**: The LLM generates an answer based on the context

```typescript
const relevantDocs = await this.retrieveDocuments(question);
const context = relevantDocs.map((doc) => doc.pageContent).join('\n\n');
// Pass context and question to LLM...
```

## 🧩 Key Components Explained

### Vector Store (HNSWLib)
- An in-memory vector database
- Uses HNSW (Hierarchical Navigable Small World) algorithm for fast similarity search
- Perfect for prototyping and small-scale applications
- Can be saved/loaded from disk

### Embeddings
- Convert text to numerical vectors
- Similar text has similar embeddings
- Used for semantic search (finding meaning, not just keywords)

### Text Splitter
- Breaks documents into manageable chunks
- `chunkSize`: Maximum characters per chunk
- `chunkOverlap`: Shared characters between chunks to preserve context

### RAG Chain
- Combines retrieval and generation
- Uses `RunnableSequence` to create a pipeline
- Includes prompt template, LLM, and output parser

## 📝 Adding Your Own Documents

1. Create `.txt` files in the `documents/` directory
2. Run the application again
3. The system will automatically:
   - Load your new documents
   - Create embeddings
   - Store them in the vector database

## 🎨 Customization Options

### Change the LLM Model
```typescript
this.llm = new OpenAI({
  modelName: 'gpt-4',  // Use GPT-4 instead
  temperature: 0.7,
});
```

### Adjust Retrieval
```typescript
// Retrieve more or fewer documents
const results = await this.vectorStore.similaritySearch(query, 5); // k=5
```

### Modify Chunk Size
```typescript
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,    // Larger chunks
  chunkOverlap: 100,  // More overlap
});
```

## 🔧 Advanced Features to Explore

Once you understand the basics, try:

1. **Different Vector Stores**: Chroma, Pinecone, Weaviate
2. **Metadata Filtering**: Filter documents by attributes
3. **Multi-Query Retrieval**: Generate multiple queries for better results
4. **Reranking**: Improve retrieval with reranking models
5. **Streaming Responses**: Stream LLM output in real-time
6. **Memory**: Add conversation history to the RAG system

## 📚 LangChain Concepts Demonstrated

- **Documents**: Base units of text with metadata
- **Text Splitters**: Break text into chunks
- **Embeddings**: Convert text to vectors
- **Vector Stores**: Store and retrieve embeddings
- **Chains**: Combine multiple operations
- **Runnables**: New LangChain execution paradigm
- **Prompts**: Template-based prompt construction
- **Output Parsers**: Process LLM responses

📖 **For detailed explanations, see [CONCEPTS.md](./CONCEPTS.md)**

## 🎓 Learning Path

Follow this order to build your understanding:

1. **Start with `index.ts`** (`npm start`)
   - Understand the basic RAG flow
   - See how documents are loaded, split, and indexed
   - Learn the retrieve → augment → generate pattern

2. **Try `interactive.ts`** (`npm run interactive`)
   - Experiment with your own questions
   - See how the same system handles different queries
   - Toggle verbose mode to see retrieval details

3. **Explore `advanced-retrieval.ts`** (`npm run advanced`)
   - Learn about similarity scores
   - Understand threshold-based filtering
   - See how query formulation affects results

4. **Study `custom-retrieval.ts`** (`npm run custom`)
   - Query expansion for better recall
   - Reranking for improved precision
   - Metadata filtering for targeted search
   - Ensemble methods for robust retrieval

5. **Master `streaming.ts`** (`npm run streaming`)
   - Token-by-token response generation
   - Real-time metrics and monitoring
   - Better UX for production applications

6. **Read `CONCEPTS.md`**
   - Deep dive into each LangChain component
   - Best practices and optimization tips
   - Common patterns and debugging techniques

## 🐛 Troubleshooting

### "Vector store not initialized"
Make sure to call `createVectorStore()` before querying.

### "OpenAI API key not found"
Check that your `.env` file has the correct API key.

### "Module not found" errors
Run `npm install` to ensure all dependencies are installed.

## 📖 Additional Resources

- [LangChain JS/TS Documentation](https://js.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [RAG Explained](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

## 🤝 Next Steps

1. Experiment with different documents
2. Try different embedding models
3. Implement a chat interface
4. Add a web UI with Express or Next.js
5. Deploy to production with a persistent vector database

## 📄 License

MIT

---

Happy learning! 🎉
# langchain-rag
