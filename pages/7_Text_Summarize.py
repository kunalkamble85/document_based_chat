import streamlit as st
from utils.langchain_utils import summarize_document, display_sidebar
from utils.file_reader import *


st.set_page_config(page_title="🤖 Text Summarize", page_icon=":book:", layout="wide")
display_sidebar()
st.title("🤖 Text Summarize")

how_select = st.radio("How would you like provide your text?", ["Upload File", "Enter Text"], horizontal=True)

max_tokens = st.slider("Number of words", min_value=100, max_value=4000, value=200, step=100)

if how_select == "Enter Text":
    code_text = st.text_area("Enter your text to summarize", height= 200)
else:
    uploaded_file = st.file_uploader(label="Choose a file")

button = st.button(label=f"Summarize")


if button:
    with st.spinner('Processing...'):
        if how_select == "Enter Text" and code_text:
            text = code_text
        elif how_select == "Upload File" and uploaded_file:
            try:
                file_name = uploaded_file.name
                if uploaded_file.type == "application/pdf":
                    raw_text = get_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    raw_text = get_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    raw_text = get_text_from_txt(uploaded_file)
                elif uploaded_file.type == "text/csv":
                    raw_text = get_text_from_csv(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    raw_text = get_text_from_xlsx(uploaded_file)
                else:
                    raw_text = get_text_from_txt(uploaded_file)                                                    
                st.info(f"File {file_name} uploaded successfully.")
                text = raw_text
            except Exception as e:
                st.error(f"Error while uploading File {file_name} : {e}")
                print(e)
        if text!=None and text!="":
            output = summarize_document(text, max_tokens)
            st.write("Click below button to download summary generated.")
            st.download_button('Download Summary', output, "summary.txt")            
            st.success('Successfully processed. Below is the required summary.')
            st.markdown(f"""<p style="background-color: #DEDDDD">{output}</p>""", unsafe_allow_html=True)
        else:
            st.error('Error while processing input!')