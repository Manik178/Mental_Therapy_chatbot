import os
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

def get_pdf_paths(directory="/Users/kartikaydev/Desktop/AIproject/Mental_Therapy_chatbot/Lang_approach/src/doc_pdf"):
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

def get_conversational_chain(vector_store):
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

def main():
    # 1. Build or load index
    if not os.path.isdir(INDEX_DIR):
        print("Building FAISS index from PDFs…")
        pdfs = get_pdf_paths()
        text = get_pdf_text(pdfs)
        chunks = get_text_chunks(text)
        build_faiss_index(chunks)
    else:
        print("Loading existing FAISS index…")

    vector_store = load_faiss_index()
    chain = get_conversational_chain(vector_store)

    # 2. REPL loop
    print("Enter questions (type 'exit' to quit):")
    while True:
        q = input("▶ ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        response = chain.invoke({"query": q})
        print("\nAnswer ▶", response["result"], "\n")


if __name__ == "__main__":
    main()
