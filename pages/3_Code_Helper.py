import streamlit as st
from utils.langchain_utils import process_source_code


st.set_page_config(page_title="🤖 Code conversion/Documentation", page_icon=":book:", layout="wide")
st.title("🤖 Code Conversion/Documentation")

task = st.radio("What would you like to do?", ["Convert Code", "Explain Code"], horizontal=True)

source_lang = st.selectbox("Source Language", ("COBOL", "C","Java","C++","PHP","Python",".NET","R","JavaScript"))

target_lang = None
if task == "Convert Code":
    target_lang = st.selectbox("Target Language", ("Python", "COBOL", "C","Java","C++","PHP",".NET","R","JavaScript"))

sample_data = open("./sample_data/cobol_code_sample.txt").read()
st.download_button('Download sample file', sample_data, "cobol_code_sample.txt")

how_select = st.radio("How would you like to do?", ["Piece of Code", "Upload Source File"], horizontal=True)

if how_select == "Piece of Code":
    code_text = st.text_area("Insert your code", height= 200)
else:
    documents = st.file_uploader(label="Choose a source file")

button = st.button(label=f"{task}")

if button:
    with st.spinner('Processing...'):
        if how_select == "Piece of Code" and code_text:
            code =  code_text
        elif how_select == "Upload Source File" and documents:
            code = documents.read()     
            
        output = process_source_code(code, task, source_lang, target_lang)
        st.write("Click below button to download test cases generated.")
        st.download_button('Download output', output, "output.txt")
        st.success('Successfully processed!')