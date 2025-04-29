import os
import fitz  # PyMuPDF
import docx

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_documents(directory="Data/Documents"):
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            ext = file.lower().split('.')[-1]
            try:
                if ext == 'txt':
                    text = read_txt(path)
                elif ext == 'pdf':
                    text = read_pdf(path)
                elif ext == 'docx':
                    text = read_docx(path)
                else:
                    continue
                docs.append({'text': text, 'path': path})
            except Exception as e:
                print(f"Failed to read {file}: {e}")
    return docs
