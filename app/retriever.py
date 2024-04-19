import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.openai_chat import OpenAIChat

# Load environment variables from .env file
load_dotenv()
openai_chat = OpenAIChat()
persist_directory = "db, "
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def build_retriever(document_path):
    loader = Docx2txtLoader(document_path)
    data = loader.load()
    # print(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    # print(all_splits)
    # Store splits

    vector_db = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    vector_db.persist()
    return vector_db.as_retriever()


def search_retriever(retriever, user_query):
    return retriever.similarity_search(user_query)
