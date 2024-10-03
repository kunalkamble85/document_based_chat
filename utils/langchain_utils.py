from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import traceback
import tempfile
import os
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import pandas as pd
import streamlit as st
import kuzu
from utils.oci_utils import *

def get_test_case_for_code(user_input):
    template = f"""
        You are an expert programmer and you know how to write test cases in any programming languages. 
        In user_input, I would be providing you the source code function. 
        Analyze the source code and generate test cases in same programming language.  
        Do not include my question in your answer.
        {user_input}
        """
    return template

def get_test_case_for_use_case(user_input):
    template = f"""
        You are an expert programmer and you know how to write test cases in any programming languages. 
        In user_input, I would be providing you the business use case. 
        Understand the context of use case and generate higher level test cases.  
        Do not include my question in your answer.
        {user_input}
        """
    return template

def get_generate_user_stories_prompt(user_input):
    template = f"""
        You are an certified product owner of Agile Scrum methodology and you know how to write perfect User Story for Agile Scrum process. 
        In user_input, I would be providing you the Buiness Use Case. 
        Understand the context of use case and generate detailed level user stories.
        Output format must be json object with Summary, Who, What, Why and Acceptance_Criteria tags.
        Buiness Use Case:
        {user_input}
        """
    return template


def get_prompt_for_code_conversion(input, source_language, target_language):
    template = f"""
        You are an expert programmer of {source_language} and {target_language} programming languages. 
        In user_input, I would be providing you the source code. 
        Analyze the {source_language} source code and convert the code into {target_language}.  
        Generate proper comments for the code. 
        Generate the unit test cases for the code generated.
        {input}
        """
    return template

def get_prompt_for_code_generation(input, source_language):
    template = f"""
        You are an expert programmer of {source_language}. 
        Generate code in {source_language} language for below input.  
        Generate proper comments for the code. 
        Generate the unit test cases for the code generated.
        {input}
        """
    return template

def get_prompt_for_code_explain(input, source_language):
    template = f"""
        You are an expert programmer of {source_language} programming language. 
        In user_input, I would be providing you the source code. 
        Analyze the {source_language} source code and explain the code line by line.
        do not include code into the explaination.
        {input}
        """
    return template

def get_prompt_summary_task(input, max_tokens):
    template = f"""
        You are helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        Use the given text below and generate the summary with {max_tokens} words.
        {input}
        """
    return template

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
    
def get_text_from_documents(file):
    text = ""
    try:
        file_extension = os.path.splitext(file.name)[1]
        print(file_extension)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        elif file_extension == ".doc" or file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        if loader:
            for doc in loader.load():
                text += "\n" + doc.page_content
            os.remove(temp_file_path)
            print(f"Got the content: {file}")
    except:
        print(traceback.format_exc())
        print(f"Error while loading file: {file}")    
    return text


def generate_tests_using_google(input, option):
    if option == "Code":
        prompt = get_test_case_for_code(input)
    else:
        prompt = get_test_case_for_use_case(input)
    output = generate_oci_gen_ai_response(st.session_state.LLM_MODEL, [{"role":"user", "content": prompt}])
    return output

def generate_user_stories(input):
    prompt = get_generate_user_stories_prompt(input)
    output = generate_oci_gen_ai_response(st.session_state.LLM_MODEL, [{"role":"user", "content": prompt}])
    return output


def process_source_code(input, option, source_language, target_language):
    if option == "Convert Code":
        prompt = get_prompt_for_code_conversion(input, source_language, target_language)
    elif(option == "Explain Code"):
        prompt = get_prompt_for_code_explain(input, source_language)
    else:
        prompt = get_prompt_for_code_generation(input, source_language)
    
    response = generate_oci_gen_ai_response(st.session_state.LLM_MODEL, [{"role":"user", "content": prompt}])
    response = response.replace("```","") 
    if target_language is not None:
        response = response.replace(target_language,"")
    return response


def summarize_document(text, max_tokens):
    prompt = get_prompt_summary_task(text, max_tokens)
    response = generate_oci_gen_ai_response(st.session_state.LLM_MODEL, [{"role":"user", "content": prompt}])
    return response

def generate_synthetic_data(num_of_records):
    prompt = f"""
    You an Asset Manager, and you need to generate fake data which look like real data for benchamark names.
    Generate unique and distinctive benchmark names like S&P 500 Index and S&P 500 Growth Index. 
    Generate {num_of_records+5} names like that and do not give any category to it"""
    response = generate_oci_gen_ai_response(st.session_state.LLM_MODEL, [{"role":"user", "content": prompt}])
    return response

def insert_graph_entries(lines):
    conn = kuzu.Connection(st.session_state.kuzu_database)
    for sql in lines:
        try:
            print(f"Executing:{sql}")
            conn.execute(sql)
        except:
            print(f"error executing:{sql}")


def create_graph_in_database(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    with open(temp_file_path) as file:
        lines = file.readlines()
        insert_graph_entries(lines)
