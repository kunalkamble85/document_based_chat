import streamlit as st
from utils.langchain_utils import process_source_code

source = {"python":"Python", "java":"Java", "c":"C", "cobol":"COBOL", "php":"PHP","r":"R","js":"JavaScript","angularJS":"angularJS","reactJS":"reactJS"}
target = {"java":"Java","python":"Python","php":"PHP","r":"R","js":"JavaScript", "reactJS":"reactJS", "angularJS":"angularJS"}
languages_extensions = {"java":".java","php":".php","python":".py","r":".R","js":".js", "c":".c", "cobol":".txt", "reactJS":".js","angularJS":".js"}


st.set_page_config(page_title="🤖 Code conversion/Documentation", page_icon=":book:", layout="wide")
st.title("🤖 Code Conversion/Documentation")

task = st.radio("What would you like to do?", ["Generate Code", "Convert Code", "Explain Code"], horizontal=True)

source_lang = st.selectbox("Source Language", source.keys(), format_func=lambda x: source[x])

code_text =  None
documents = None
how_select = None
input_text = None
target_lang = None
if task == "Generate Code":
    input_text = st.text_area(label="Insert your requirment here to generate code", value="write a program to sort list of numbers", height= 200)
    target_lang = source_lang
else:    
    if task == "Convert Code":
        target_lang = st.selectbox("Target Language", target.keys(), format_func=lambda x: target[x])

    sample_data = open("./sample_data/cobol_code_sample.txt").read()
    st.download_button('Download sample file', sample_data, "cobol_code_sample.txt")

    how_select = st.radio("How would you like to do?", ["Upload Source File","Piece of Code", ], horizontal=True)

    if how_select == "Piece of Code":
        code_text = st.text_area("Insert your code", value="print('Hello')", height= 200)
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
        if task == "Generate Code":
            code =  input_text
            
        output = process_source_code(code, task, source_lang, target_lang)
        st.write("Click below button to download test cases generated.")
        print(f"file_name:{file_name}")
        if task == "Explain Code":
            target_file_name = "code_documentation.txt"
        else:
            target_file_name = f"{file_name.split('.')[0]}{languages_extensions.get(target_lang)}"        
        st.download_button('Download output', output, target_file_name)
        st.success('Successfully processed. Below is the code snippet.')
        st.code(output, language=target_lang, line_numbers=False)