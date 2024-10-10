import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

def init(data_dir='rag_data'):
    """
    Initialize the RAG system by loading documents and creating a query engine.
    
    Args:
    data_dir (str): Directory containing the documents to be indexed.
    
    Returns:
    query_engine: A query engine object for performing queries.
    """
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    # Create a vector store index
    index = VectorStoreIndex.from_documents(documents)
    
    # Create and return a query engine
    return index.as_query_engine()

def query(query_engine, question):
    """
    Perform a query using the provided query engine.
    
    Args:
    query_engine: The query engine to use for the query.
    question (str): The question to ask.
    
    Returns:
    str: The response to the query.
    """
    # Perform the query
    response = query_engine.query(question)
    
    return str(response)

# Example usage:
if __name__ == "__main__":
    # Initialize the system
    query_engine = init()
    
    # Perform a query
    question = "What is RAG?"
    answer = query(query_engine, question)
    print(answer)
