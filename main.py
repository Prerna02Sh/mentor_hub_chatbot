from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# -----------------------
# 🔐 Load API Keys
# -----------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_API_TOKEN")

# -----------------------
# ⚙️ FastAPI Setup
# -----------------------
app = FastAPI(title="PDF Chatbot API")

# -----------------------
# 📄 PDF Retriever
# -----------------------
def load_and_prepare_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# -----------------------
# 🧠 Prompt Template
# -----------------------
def get_custom_prompt():
    template = """You are a helpful assistant which is based on mentorhub data. Use the following context from the PDF to answer the user's question.
If you don't know the answer, say "I’m not sure based on the PDF content." and help the user politely and effectively.

Context:
{context}

Question:
{question}

Answer (be concise and friendly):"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt

# -----------------------
# 🔗 Load LLM Chain
# -----------------------
def get_chain():
    retriever = load_and_prepare_retriever("dataset.pdf")
    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
    prompt = get_custom_prompt()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return chain

# -----------------------
# 📦 Load Chain on Startup
# -----------------------
qa_chain = get_chain()

# -----------------------
# 📥 Request Schema
# -----------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------
# 📤 POST Endpoint
# -----------------------
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        response = qa_chain.run(request.question)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
