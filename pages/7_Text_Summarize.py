import streamlit as st
from utils.langchain_utils import get_text_from_documents, summarize_document, display_sidebar


st.set_page_config(page_title="🤖 Text Summarize", page_icon=":book:", layout="wide")
display_sidebar()
st.title("🤖 Text Summarize")

how_select = st.radio("How would you like provide your text?", ["Upload File", "Enter Text"], horizontal=True)

max_tokens = st.slider("Number of words", min_value=100, max_value=4000, value=200, step=100)

if how_select == "Enter Text":
    code_text = st.text_area("Enter your text to summarize", height= 200)
else:
    documents = st.file_uploader(label="Choose a file", type=["txt","doc","pdf"])

button = st.button(label=f"Summarize")


if button:
    with st.spinner('Processing...'):
        if how_select == "Enter Text" and code_text:
            text = code_text
        elif how_select == "Upload File" and documents:
            text = get_text_from_documents(documents)     
        
        if text!=None and text!="":
            output = summarize_document(text, max_tokens)
            st.write("Click below button to download summary generated.")
            st.download_button('Download Summary', output, "summary.txt")            
            st.success('Successfully processed. Below is the required summary.')
            st.markdown(f"""<p style="background-color: #DEDDDD">{output}</p>""", unsafe_allow_html=True)
        else:
            st.error('Error while processing input!')