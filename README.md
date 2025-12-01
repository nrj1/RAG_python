# RAG Pipeline Implementation

## Overview
A Retrieval-Augmented Generation (RAG) pipeline that enables AI models to retrieve and use information from custom documents before generating responses. This implementation focuses on efficient document processing, chunking, and retrieval mechanisms.

## What is RAG?
RAG (Retrieval-Augmented Generation) combines information retrieval with language generation. It allows AI models to search through your documents and use that context to provide more accurate, grounded responses—essentially giving the model a searchable memory.

## Key Features
- **Efficient Document Loading**: Uses TextLoader for simple text files
- **Smart Chunking**: Implements RecursiveCharacterTextSplitter for handling small chunk sizes
- **Vector Storage**: Consistent API usage with langchain_community.vectorstores.Chroma
- **Similarity Search**: Cosine similarity for effective document retrieval
- **End-to-end Pipeline**: Complete ingestion and retrieval workflow

## Technical Components

### 1. Document Ingestion
- **Loader**: TextLoader (instead of UnstructuredLoader for .txt files)
- **Splitter**: RecursiveCharacterTextSplitter
- **Vector Store**: langchain_community.vectorstores.Chroma

### 2. Document Retrieval
- **Similarity Metric**: Cosine similarity
- **Vector Store**: langchain_community.vectorstores.Chroma (consistent with ingestion)

## Installation

```bash
pip install langchain langchain-community chromadb
```

## Usage

### Basic Implementation

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# Load documents
loader = TextLoader("path/to/your/file.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Retrieve relevant documents
query = "Your search query here"
results = vectorstore.similarity_search(query, k=3)
```

## Lessons Learned

### 1. TextLoader vs UnstructuredLoader
DirectoryLoader defaults to UnstructuredLoader for .txt files, which requires additional dependencies and is overkill for simple text files. Using TextLoader directly is more efficient.

### 2. RecursiveCharacterTextSplitter vs CharacterTextSplitter
For small chunk sizes, RecursiveCharacterTextSplitter performs significantly better than CharacterTextSplitter. It handles text splitting more intelligently by trying multiple separators.

### 3. API Consistency
Ensure both ingestion and retrieval use the same Chroma implementation:
- ✅ Use: `langchain_community.vectorstores.Chroma`
- ❌ Avoid mixing with: `langchain_chroma.Chroma`

Different implementations have different APIs and can cause retrieval failures.

## Project Status
🚧 **Under Development** - This RAG pipeline is being integrated into an upcoming project.

## Future Enhancements
- [ ] Support for multiple document formats (PDF, DOCX)
- [ ] Hybrid search (keyword + semantic)
- [ ] Query optimization
- [ ] Performance benchmarking
- [ ] Caching mechanism

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
MIT License

## Contact
Connect with me on [LinkedIn](https://www.linkedin.com/in/neil-john/)

---

*Built with hands-on learning and debugging. Every error was a lesson.*
