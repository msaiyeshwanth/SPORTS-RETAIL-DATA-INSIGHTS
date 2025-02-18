import os
import faiss
import pandas as pd
import threading
from time import sleep
from IPython.display import display, clear_output
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from ipywidgets import FileUpload, VBox, Output, Textarea, Button

# Global Variables
vector_store = None
llm = None

# UI Elements
upload_widget = FileUpload(multiple=True)
chat_input = Textarea(placeholder="Ask a question...", layout={'width': '100%', 'height': '50px'})
send_button = Button(description="Submit")
output = Output()

# Loading Animation
loading = False

def show_loading():
    global loading
    with output:
        while loading:
            clear_output(wait=True)
            print("⏳ Processing your question...")
            sleep(0.5)

def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        filename = file["metadata"]["name"]
        filepath = f"/mnt/data/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(file["content"])
        
        ext = filename.split(".")[-1].lower()
        
        if ext == "pdf":
            loader = PyPDFLoader(filepath)
        elif ext in ["docx", "doc"]:
            loader = Docx2txtLoader(filepath)
        elif ext == "txt":
            loader = TextLoader(filepath)
        elif ext in ["csv", "xlsx"]:
            df = pd.read_excel(filepath) if ext == "xlsx" else pd.read_csv(filepath)
            text_data = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
            docs.append(text_data)
            continue
        else:
            print(f"Unsupported file format: {filename}")
            continue
        
        docs.extend(loader.load())
    
    return docs

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    
    return vector_store

def load_local_llm():
    return CTransformers(
        model="TheBloke/Mistral-7B-Instruct-GGUF",
        model_type="mistral",
        config={"max_new_tokens": 500, "temperature": 0.3}
    )

def generate_answer(vector_store, llm, question):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    result = qa_chain({"query": question})
    
    return result["result"]

def process_upload(change):
    global vector_store, llm
    with output:
        output.clear_output()
        print("🔄 Processing documents...")

        documents = load_documents(upload_widget.value.values())
        vector_store = create_vector_store(documents)

        llm = load_local_llm()
        print("✅ Documents processed successfully. You can now ask questions.")

upload_widget.observe(process_upload, names="_counter")

def on_submit(b):
    global loading
    if vector_store is None:
        with output:
            output.clear_output()
            print("⚠️ Please upload documents first.")
        return
    
    question = chat_input.value.strip()
    if not question:
        return

    loading = True
    loading_thread = threading.Thread(target=show_loading)
    loading_thread.start()

    answer = generate_answer(vector_store, llm, question)

    loading = False
    loading_thread.join()

    with output:
        clear_output(wait=True)
        print(f"**You:** {question}\n")
        print(f"**AI:** {answer}")

    chat_input.value = ""

send_button.on_click(on_submit)

# Display Chatbot UI
display(VBox([upload_widget, chat_input, send_button, output]))
