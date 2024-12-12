from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path relative to the current directory
data_path = os.path.join(base_dir, "data")



def load_documents(file_path = data_path) -> List[Document]:
    documents = []
    for file_name in os.listdir(file_path):
        file_path = os.path.join(file_path,file_name)
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print("Unsupported file type ")
            continue
        documents.extend(loader.load())
    print(len(documents))
    
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200,
        length_function = len
    )

    splits = text_splitter.split_documents(documents)
    print("The no of splits is",len(splits))
    return splits

def emmbed_documents(chunks):
    embbeding = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004")
    document_embedding = embbeding.embed_documents([chunks.page_content for chunks in chunks])
    print(f"Created embeddings for {len(document_embedding)} document chunks.")
    return document_embedding
    
    
