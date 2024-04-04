from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
import traceback
import tempfile
import os
from langchain_core.prompts import (PromptTemplate)
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import pandas as pd
import streamlit as st
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatGooglePalm

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


def get_test_case_for_code(user_input):
    template = f"""
        <s>[INST]
        You are an expert programmer and you know how to write test cases in any programming languages. 
        In user_input, I would be providing you the source code function. 
        Analyze the source code and generate test cases in same programming language.  
        Do not include my question in your answer.
        {user_input}
        [/INST]
        """
    return template

def get_test_case_for_use_case(user_input):
    template = f"""
        <s>[INST]
        You are an expert programmer and you know how to write test cases in any programming languages. 
        In user_input, I would be providing you the business use case. 
        Understand the context of use case and generate higher level test cases.  
        Do not include my question in your answer.
        {user_input}
        [/INST]
        """
    return template

def clear_vectordb(embedding_function):
    vector_store = Chroma(persist_directory=st.session_state.CHROMA_DB_PATH, embedding_function = embedding_function)
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
        st.session_state.vector_store = Chroma.from_documents(persist_directory=st.session_state.CHROMA_DB_PATH, documents = chunks, embedding=embedding_function)

import requests

def get_test_cases(input, option):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}

    if option == "Code":
        prompt = get_test_case_for_code(input)
    else:
        prompt = get_test_case_for_use_case(input)
    
    print(prompt)
    
    payload = {
                "inputs": prompt,
                "parameters": {"temperature": 0.8}
            }
    response = requests.post(API_URL, headers=headers, json=payload)
    output = response.json()   
    print("---------------------------------------------------")
    print(output)
    print("---------------------------------------------------")
    text = output[0]["generated_text"]
    if "```" in text:
        return "\t" + text.split("```")[-1].strip()
    elif "[/INST]" in text:
        return "\t" + text.split("[/INST]")[-1].strip()
    return text

def generate_tests_using_google(input, option):
    if option == "Code":
        prompt = get_test_case_for_code(input)
    else:
        prompt = get_test_case_for_use_case(input)
    llm = ChatGooglePalm(temprature = 0.5, model_kwargs={"max_length": 200})
    qa_chain = ConversationChain(llm=llm)
    output = qa_chain.invoke(prompt)
    print(output)
    if "response" in output:
        return output["response"]
    return "Not able to fetch test cases."
