import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 環境変数の読み込み
load_dotenv()

def create_vectorstore_from_pdf(pdf_paths):
    """
    PDFファイルのリストを読み込み、ベクトルストアを作成して返す
    """
    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # OpenAI Embeddingsを使用
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore

def get_rag_chain(vectorstore):
    """
    ベクトルストアを受け取り、RAGチェーンを作成して返す
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "あなたは質問回答のアシスタントです。"
        "以下の取得されたコンテキストを使用して、質問に答えてください。"
        "答えがわからない場合は、わからないと答えてください。"
        "回答は簡潔に保ってください。"
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
