from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def retrieval_pipeline(query: str):
    vectorstore = Chroma(
        persist_directory="db/chroma_db", 
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"), 
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever.invoke(query)

def format_results(query: str, results):
    """Format retrieval results in a readable way."""
    output = []
    output.append("=" * 80)
    output.append(f"QUERY: {query}")
    output.append("=" * 80)
    output.append(f"\nRetrieved {len(results)} document(s):\n")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        # Extract just the filename from the path
        filename = source.split('/')[-1] if '/' in source else source
        output.append(f"--- Document {i} (Source: {filename}) ---")
        output.append(f"{doc.page_content}")
        output.append("")  # Empty line for spacing
    
    return "\n".join(output)

if __name__ == "__main__":
    queries = [
        "What famous phrase did Oliver Twist say that led to him being sold as an apprentice?",
        "What physical disability did Long John Silver have in Treasure Island?",
        "What was the name of Jane Eyre's cousin who proposed to her?",
        "What was the name of the dog that accompanied the three men on their boating trip?",
        "What happened to Mr. Rochester's wife Bertha at the end of Jane Eyre?",
        "What was the name of the ship in Treasure Island?",
        "Who was the compassionate woman in Oliver Twist who tried to help Oliver but was murdered for her betrayal?",
        "What river did the three men travel on during their boating holiday?",
    ]

    for query in queries:
        results = retrieval_pipeline(query)
        print(format_results(query, results))
        print("\n")  # Extra spacing between queries
