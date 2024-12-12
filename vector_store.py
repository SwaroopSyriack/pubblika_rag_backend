from langchain_community.vectorstores import Chroma
from load_emmbed import load_documents,split_documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

load = load_documents()
chunks = split_documents(load)


def vectorize():
    collection_name = "my_collection"
    vectorstore = Chroma.from_documents(
        collection_name=collection_name,
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="./chroma_db"
    )
    print("Vector store created and persisted to './chroma_db'")

    return vectorstore


