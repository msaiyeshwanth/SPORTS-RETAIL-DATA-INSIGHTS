import os
import faiss
import pandas as pd
import threading
from time import sleep
from pptx import Presentation
from email import policy
from email.parser import BytesParser
from IPython.display import display, clear_output
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from ipywidgets import FileUpload, VBox, Output, Textarea, Button

# ---------------------------------
# 1️⃣ Upload Documents (Multiple Formats)
# ---------------------------------

upload_widget = FileUpload(multiple=True)
output = Output()

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a directory."""
    saved_files = []
    for file in uploaded_files:
        filename = file["metadata"]["name"]
        filepath = f"/mnt/data/{filename}"

        with open(filepath, "wb") as f:
            f.write(file["content"])

        saved_files.append(filepath)
    return saved_files

display(VBox([upload_widget, output]))

# ---------------------------------
# 2️⃣ Process & Extract Content from Files
# ---------------------------------

def extract_ppt_text(filepath):
    """Extract text from PowerPoint (PPTX)."""
    prs = Presentation(filepath)
    text_data = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_data.append(shape.text)
    return "\n".join(text_data)

def extract_email_text(filepath):
    """Extract content from an email file (EML)."""
    with open(filepath, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg.get_body(preferencelist=("plain", "html")).get_content()

def load_documents(file_paths):
    """Load and process documents from different formats."""
    documents = []
    
    for filepath in file_paths:
        ext = filepath.split(".")[-1].lower()
        
        if ext == "pdf":
            loader = PyPDFLoader(filepath)
        elif ext in ["docx", "doc"]:
            loader = Docx2txtLoader(filepath)
        elif ext == "txt":
            loader = TextLoader(filepath)
        elif ext in ["csv", "xlsx"]:
            df = pd.read_excel(filepath) if ext == "xlsx" else pd.read_csv(filepath)
            text_data = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
            documents.append(text_data)
            continue
        elif ext == "pptx":
            documents.append(extract_ppt_text(filepath))
            continue
        elif ext == "eml":
            documents.append(extract_email_text(filepath))
            continue
        else:
            with output:
                output.clear_output()
                print(f"❌ Unsupported file format: {filepath}")
            continue
        
        documents.extend(loader.load())
    
    return documents

def create_vector_store(documents):
    """Create a FAISS vector store with embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    
    return vector_store

# ---------------------------------
# 3️⃣ Load Local LLM Model (Mistral-7B, Phi-2)
# ---------------------------------

def load_local_llm():
    """Load a lightweight LLM for text generation."""
    return CTransformers(
        model="TheBloke/Mistral-7B-Instruct-GGUF",
        model_type="mistral",
        config={"max_new_tokens": 500, "temperature": 0.3}
    )

vector_store = None
llm = None

def process_upload(change):
    """Handle document upload and processing."""
    global vector_store, llm
    with output:
        output.clear_output()
        print("🔄 Processing uploaded documents...")

    file_paths = save_uploaded_files(upload_widget.value.values())
    documents = load_documents(file_paths)

    if not documents:
        with output:
            output.clear_output()
            print("⚠️ No valid documents found. Please upload supported files.")
        return

    vector_store = create_vector_store(documents)
    llm = load_local_llm()

    with output:
        output.clear_output()
        print("✅ Documents processed successfully. You can now ask questions.")

upload_widget.observe(process_upload, names="_counter")

# ---------------------------------
# 4️⃣ Interactive Chatbot
# ---------------------------------

chat_input = Textarea(placeholder="Ask a question...", layout={'width': '100%', 'height': '50px'})
send_button = Button(description="Submit")
chat_output = Output()

loading = False

def show_loading():
    """Display a loading message while processing the question."""
    global loading
    with chat_output:
        while loading:
            clear_output(wait=True)
            print("⏳ Processing your question...")
            sleep(0.5)

def generate_answer(question):
    """Retrieve answer from vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    result = qa_chain({"query": question})
    return result["result"]

def on_submit(b):
    """Handle user question submission."""
    global loading
    if vector_store is None:
        with chat_output:
            chat_output.clear_output()
            print("⚠️ Please upload and process documents first.")
        return

    question = chat_input.value.strip()
    if not question:
        return

    loading = True
    loading_thread = threading.Thread(target=show_loading)
    loading_thread.start()

    answer = generate_answer(question)

    loading = False
    loading_thread.join()

    with chat_output:
        clear_output(wait=True)
        print(f"**You:** {question}\n")
        print(f"**AI:** {answer}")

    chat_input.value = ""

send_button.on_click(on_submit)

# Display Chatbot UI
display(VBox([chat_input, send_button, chat_output]))
