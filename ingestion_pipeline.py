import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(directory: str):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Small chunk size for testing (in characters)
        chunk_overlap=10,
        separators=[""]  # Only character-level splitting - no word/paragraph boundaries
    )
    chunks = text_splitter.split_documents(documents)
    '''
    print(f"Split {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
        print(f"\n--- Chunk {i+1} ({len(chunk.page_content)} chars) ---")
        print(chunk.page_content)'''
    
    return chunks

def vector_store(chunks, persist_directory: str = "db/chroma_db"):
    print("Creating vector store...")
    
    # OpenAIEmbeddings automatically reads OPENAI_API_KEY from environment variables
    # Make sure you have OPENAI_API_KEY set in your .env file or environment
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created and persisted to {persist_directory}")
    return vectorstore

def main():
    print("Starting ingestion pipeline...")
    documents = load_documents("docs")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    vectorstore = vector_store(chunks, persist_directory="db/chroma_db")
    print("Ingestion pipeline completed successfully!")

if __name__ == "__main__":
    main()
   