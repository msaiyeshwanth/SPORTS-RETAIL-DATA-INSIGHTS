!pip install transformers langchain sentence-transformers faiss-cpu pypdf python-docx python-pptx openpyxl email


import PyPDF2
import os
import glob
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from IPython.display import display
from ipywidgets import FileUpload, Text
import docx
from pptx import Presentation
import openpyxl
from email import policy
from email.parser import BytesParser



def extract_text_from_pdfs(pdf_folder):
    text_data = []
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            text_data.append(text)
    return text_data

def extract_text_from_word(doc_folder):
    text_data = []
    doc_files = glob.glob(os.path.join(doc_folder, "*.docx"))
    for doc_file in doc_files:
        doc = docx.Document(doc_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        text_data.append(text)
    return text_data

def extract_text_from_txt(txt_folder):
    text_data = []
    txt_files = glob.glob(os.path.join(txt_folder, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            text = file.read()
            text_data.append(text)
    return text_data

def extract_text_from_ppt(ppt_folder):
    text_data = []
    ppt_files = glob.glob(os.path.join(ppt_folder, "*.pptx"))
    for ppt_file in ppt_files:
        presentation = Presentation(ppt_file)
        text = "\n".join([slide.shapes.text for slide in presentation.slides if hasattr(slide.shapes, 'text')])
        text_data.append(text)
    return text_data

def extract_text_from_excel(excel_folder):
    text_data = []
    excel_files = glob.glob(os.path.join(excel_folder, "*.xlsx"))
    for excel_file in excel_files:
        wb = openpyxl.load_workbook(excel_file)
        sheet = wb.active
        text = "\n".join([str(cell.value) for row in sheet.iter_rows() for cell in row])
        text_data.append(text)
    return text_data

def extract_text_from_emails(email_folder):
    text_data = []
    email_files = glob.glob(os.path.join(email_folder, "*.eml"))
    for email_file in email_files:
        with open(email_file, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
            text = msg.get_body(preferencelist=('plain')).get_payload()
            text_data.append(text)
    return text_data





upload_widget = FileUpload(accept=".pdf,.docx,.txt,.pptx,.xlsx,.eml", multiple=True)
display(upload_widget)

def process_uploaded_files(upload_widget):
    # Save uploaded files
    file_folder = "uploaded_files"
    os.makedirs(file_folder, exist_ok=True)
    
    for filename, file in upload_widget.value.items():
        with open(os.path.join(file_folder, filename), 'wb') as f:
            f.write(file['content'])
    
    # Extract text from different file types
    pdf_text = extract_text_from_pdfs(file_folder)
    word_text = extract_text_from_word(file_folder)
    txt_text = extract_text_from_txt(file_folder)
    ppt_text = extract_text_from_ppt(file_folder)
    excel_text = extract_text_from_excel(file_folder)
    email_text = extract_text_from_emails(file_folder)
    
    # Combine all extracted texts
    all_text = pdf_text + word_text + txt_text + ppt_text + excel_text + email_text
    return all_text

documents = process_uploaded_files(upload_widget)



splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = [chunk for doc in documents for chunk in splitter.split_text(doc)]

# Generate embeddings using SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)

# Store embeddings in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))





def retrieve_top_k(query, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# Set up the model for text generation
model = pipeline("text-generation", model="microsoft/phi-2")

def generate_answer_with_rag(query):
    # Retrieve the top-k most relevant chunks
    context = "\n".join(retrieve_top_k(query))
    
    if not context:  # Check if relevant context is found
        return "No relevant information found in the documents. Please try rephrasing your question or upload more documents."

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Generate answer using the model
    response = model(prompt, max_length=200, num_return_sequences=1)
    generated_answer = response[0]['generated_text']

    # Post-processing validation step: Cross-check if the generated answer is consistent with the retrieved context
    validation_result = validate_answer(query, generated_answer, context)
    
    if validation_result:
        return generated_answer
    else:
        return "The generated answer seems inconsistent with the provided context. Please try again with a more specific question."

def validate_answer(query, generated_answer, context):
    # Simple validation: Check if generated answer contains parts of the context
    # This can be improved with more sophisticated checks (e.g., NLP-based similarity checks)
    for chunk in context.split("\n"):
        if chunk.lower() in generated_answer.lower():
            return True
    return False



question_widget = Text(description="Ask a Question:")
display(question_widget)

def on_question_submit(change):
    question = question_widget.value
    print(f"Question: {question}")
    
    answer = generate_answer_with_rag(question)
    print(f"Answer: {answer}")
    
    # Allow user to re-ask if the answer is wrong
    response = input("Is the answer correct? (yes/no): ")
    if response.lower() == "no":
        print("Feel free to rephrase the question or upload more documents for more context.")
    
question_widget.observe(on_question_submit, names="value")
