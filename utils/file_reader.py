from docx import Document
import io
import fitz
import pandas as pd
import os

def get_text_from_txt(uploaded_file):
    try:
        text = uploaded_file.read().decode("utf-8")
        print(f"Successfully saved file {uploaded_file}")
        return text
    except Exception as e:
        return f"Error: {e}"

def get_text_from_pdf(uploaded_file):
    try:
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text =""
        for pages in document.pages():
            text += pages.get_text()
        print(f"Successfully saved file {uploaded_file}")
        return text
    except Exception as e:
        return f"Error: {e}"
    
def get_text_from_docx(uploaded_file):
    try:
        stream = io.BytesIO(uploaded_file.getvalue())
        doc = Document(stream)
        text =""
        for para in doc.paragraphs:
            text+= para.text + "\n"
        print(f"Successfully saved file {uploaded_file}")
        return text
    except Exception as e:
        return f"Error: {e}"
    
def get_text_from_csv(uploaded_file):
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        print(f"Successfully saved file {uploaded_file}")
        return df.to_string()
    except Exception as e:
        return f"Error: {e}"
    
def get_text_from_xlsx(uploaded_file):
    text = ""
    try:
        df = pd.read_excel(uploaded_file)
        for column in df.columns:
            text+= " ".join(df[column].astype(str)) + "\n"   
        print(f"Successfully saved file {uploaded_file}") 
    except Exception as e:
        return f"Error: {e}"
    return text