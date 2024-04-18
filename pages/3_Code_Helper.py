import streamlit as st
from utils.langchain_utils import process_source_code

source = {"c":"C", "cobol":"COBOL", "java":"Java","php":"PHP","python":"Python","r":"R","js":"JavaScript"}
target = {"python":"Python","java":"Java","php":"PHP","r":"R","js":"JavaScript"}
languages_extensions = {"java":".java","php":".php","python":".py","r":".R","js":".js", "c":".c", "cobol":".txt"}


st.set_page_config(page_title="🤖 Code conversion/Documentation", page_icon=":book:", layout="wide")
st.title("🤖 Code Conversion/Documentation")

task = st.radio("What would you like to do?", ["Convert Code", "Explain Code"], horizontal=True)

source_lang = st.selectbox("Source Language", source.keys(), format_func=lambda x: source[x])

target_lang = None
if task == "Convert Code":
    target_lang = st.selectbox("Target Language", target.keys(), format_func=lambda x: target[x])

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
        file_name = "output.txt"
        if how_select == "Piece of Code" and code_text:
            code =  code_text
        elif how_select == "Upload Source File" and documents:
            file_name = documents.name
            code = documents.read()     
            
        output = process_source_code(code, task, source_lang, target_lang)
        st.write("Click below button to download test cases generated.")
        print(f"file_name:{file_name}")
        target_file_name = f"{file_name.split('.')[0]}{languages_extensions.get(target_lang)}"
        st.download_button('Download output', output, target_file_name)
        st.success('Successfully processed!')