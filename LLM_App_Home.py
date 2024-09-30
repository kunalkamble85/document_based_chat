# from dotenv import load_dotenv
# load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import base64
from pathlib import Path

st.set_page_config(page_title="🤖 LLM Home", page_icon=":book:", layout="wide")

custom_css = """
<style>
    [data-testid=stSidebar] {
        background-color: #DEDDDD !important;
    }
</style>
"""
# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit app content, including the sidebar
with st.sidebar:
    pass

with st.sidebar:
    logo = f"url(data:image/png;base64,{base64.b64encode(Path('./images/oracle_logo.jpg').read_bytes()).decode()})"
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: {logo};
                background-repeat: no-repeat;
                padding-top: 150px;
                background-size: 290px 120px;
                background-position: 20px 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.write("""<div style="width:100%;text-align:left">
            <br style="font-size: 1em;"><b>Powered by</b>
            <br style="font-size: 3em; font-weight: bold;"><b><u>OCI Generative AI</u></b>    
            </div>      
            """, unsafe_allow_html=True)
      
    st.write("""<div style="width:100%;text-align:left">
            <br style="font-size: 1em;"><b>Built by</b>
            <br style="font-size: 3em; font-weight: bold;"><b><u>Finergy AI Team</u></b>    
            </div>      
            """, unsafe_allow_html=True)

st.title("🤖 Welcome to Oracle Finergy LLM Demo")

cloud_models = {"OCP":["meta.llama3.1-70b","meta.llama3-70b","cohore.command-r-plus","cohore.command-r-16k"],"GCP":['Gemini Pro'],"AWS":['Claude Sonnet'],"OpenAI":['gpt-4o-mini']}
st.session_state.LLM_MODEL = "meta.llama3.1-70b"
userid = st.text_input("Enter your user id.")
print(f"Loging sucussful:{userid}")
# cloud_provider = st.radio(
#     "Choose your model provider",
#     ["GCP", "OCP","AWS", "OpenAI"],
#     captions = ["Google Cloud Provider", "Oracle Cloud Provider", "Amazon Web Services", "Azure Cloud Provder"], horizontal=True)
cloud_provider = "OCP"
model_name = st.radio(
    "Select OCI Generative AI Model", cloud_models.get(cloud_provider), horizontal=True)

# model_name = st.selectbox(
#     'Select LLM Model',
#     ('Google Palm','Mistral', 'Google Gemini', 'Claude Sonnet', 'llama', 'ChatGPT'))

st.markdown(''':red[Only Google gemini prod and Open AI's gpt4o-mini model are avaiable right now.]''')

temprature = st.slider("Set the model temprature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

button = st.button(label="Proceed")
if button:
    st.session_state.CHROMA_DB_PATH = f"./vector_database/{userid}"
    print(st.session_state.CHROMA_DB_PATH)
    st.session_state.KUZU_DB_PATH = f"./kuzu_database/{userid}"
    print(st.session_state.KUZU_DB_PATH)
    st.session_state.LLM_MODEL = model_name
    st.success("Session created successfully, please click on application from left menu.")