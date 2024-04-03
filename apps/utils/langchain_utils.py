from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain_community.chat_models import ChatGooglePalm
from dotenv import load_dotenv
import traceback
import tempfile
import os
from langchain_core.prompts import (PromptTemplate)
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import pandas as pd


load_dotenv()

# CHROMA_DB_PATH = "./chroma_vector_database"
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



def get_prompt_template():
    template = """
        You are helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        Use the given Context and History enclosed with backticks to answer the question at the end.
        Don't try to make up an answer, if you don't know, just say that you don't know.

        Chat History:
        {chat_history}

        Context: 
        {context}

        Follow Up Input: {question}
        Standalone question or instruction:
        """
    return PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])



def clear_vecotrdb(embedding_function):
    vector_store = Chroma(embedding_function = embedding_function)
    for collection in vector_store._client.list_collections():
        ids = collection.get()['ids']
        print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
        if len(ids): collection.delete(ids)

def process_documents(file, num_rows):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    real_data = pd.read_csv(temp_file_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    print(metadata)
    print(real_data.head())
    # print(metadata.visualize())
    synthesizer = SingleTablePreset(
        metadata,
        name='FAST_ML'
    )
    synthesizer.fit(
        data=real_data
    )
    synthetic_data = synthesizer.sample(
        num_rows=num_rows
    )
    print(synthetic_data.head())
    return synthetic_data
    


def store_documents_in_database(docs):
    text = []
    for file in docs:
        try:
            file_extension = os.path.splitext(file.name)[1]
            print(file_extension)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            elif file_extension == ".doc" or file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".xls" or file_extension == ".xlsx":
                loader = UnstructuredExcelLoader(temp_file_path)
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
            print(f"Saved file to vector database: {file}")
        except:
            print(traceback.format_exc())
            print(f"Error while loading file: {file}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n"], length_function=len)
        chunks = text_splitter.split_documents(text)
        # to use own embedding
        # embeddings = HuggingFaceEmbeddings(model_name='google/flan-t5-xxl',model_kwargs={'device': 'cpu'})
        # db = st.session_state.vectordb.from_texts(texts = chunks)
        # clear_vecotrdb(embedding_function)
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma.from_documents(documents = chunks, embedding=embedding_function)