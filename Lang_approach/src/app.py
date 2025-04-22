import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

INDEX_DIR = "faiss_index"


def get_pdf_paths(directory="./doc_pdf"):
    return [os.path.join(directory, fn)
            for fn in os.listdir(directory)
            if fn.lower().endswith(".pdf")]


def get_pdf_text(paths):
    text = ""
    for p in paths:
        with open(p, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def build_faiss_index(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local(INDEX_DIR)


def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def get_conversational_chain(vector_store, temperature):
    prompt = PromptTemplate(
        template="""
You are a supportive and concise mental health assistant. Use the provided context—drawn from trusted therapeutic sources—to give brief, helpful responses based on evidence-based practices.

Rules:
1. Respond with empathy, but keep it short and focused.
2. Use CBT, mindfulness, or ACT techniques when relevant.
3. Remind the user you're not a substitute for a therapist.
4. If information is missing, say so briefly and encourage seeking professional help.
5. In crisis cases, advise immediate contact with emergency services or hotlines.

Context:
{context}

User’s Question:
{question}

Your response:
""",
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )


# Streamlit UI
st.set_page_config(page_title="Mental Therapy Chatbot", layout="wide")
st.title("🧠 Mental Health Chatbot")

with st.sidebar:
    st.header("Settings")
    depth = st.slider("Response Depth", 1, 3, 2)
    creativity = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3)

if not os.path.isdir(INDEX_DIR):
    st.info("Building FAISS index from PDFs…")
    pdfs = get_pdf_paths()
    text = get_pdf_text(pdfs)
    chunks = get_text_chunks(text)
    build_faiss_index(chunks)
    st.success("Index built successfully.")

vector_store = load_faiss_index()
chain = get_conversational_chain(vector_store, temperature=creativity)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("How can I help you today?")

if user_input:
    docs = vector_store.similarity_search(user_input, k=depth)
    response = chain.invoke({"query": user_input, "input_documents": docs})
    reply = response["result"]

    st.session_state.chat_history.append((user_input, reply))

for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)
