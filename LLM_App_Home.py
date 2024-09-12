# from dotenv import load_dotenv
# load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import base64
from pathlib import Path

st.set_page_config(page_title="🤖 LLM Home", page_icon=":book:", layout="wide")

with st.sidebar:
    logo = f"url(data:image/png;base64,{base64.b64encode(Path('./images/oracle_logo.jpg').read_bytes()).decode()})"
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: {logo};
                background-repeat: no-repeat;
                padding-top: 150px;
                background-position: 20px 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
   st.write("""<div style="width:100%;text-align:left">
            <br style="font-size: 1em;"><b>Built by</b>
            <br style="font-size: 3em; font-weight: bold;"><b><u>Finergy AI Team</u></b>    
            </div>      
            """, unsafe_allow_html=True)

st.title("🤖 Welcome to Oracle Finergy LLM Demo")

cloud_models = {"GCP":["Google Palm", 'Gemini Pro'],"OCP":["Cohore"],"AWS":['Claude Sonnet'],"Azure":['GPT-4.0']}

userid = st.text_input("Enter your user id.")
print(f"Loging sucussful:{userid}")
cloud_provider = st.radio(
    "Choose your cloud provider",
    ["GCP", "OCP","AWS", "Azure"],
    captions = ["Google Cloud Provider", "Oracle Cloud Provider", "Amazon Web Services", "Azure Cloud Provder"], horizontal=True)

model_name = st.radio(
    "Select LLM Model", cloud_models.get(cloud_provider), horizontal=True)

# model_name = st.selectbox(
#     'Select LLM Model',
#     ('Google Palm','Mistral', 'Google Gemini', 'Claude Sonnet', 'llama', 'ChatGPT'))

st.markdown(''':red[Only GCP cloud with Google Palm is avaiable right now.]''')

temprature = st.slider("Set the model temprature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

button = st.button(label="Proceed")
if button:
    st.session_state.CHROMA_DB_PATH = f"./vector_database/{userid}"
    print(st.session_state.CHROMA_DB_PATH)
    st.session_state.KUZU_DB_PATH = f"./kuzu_database/{userid}"
    print(st.session_state.KUZU_DB_PATH)
    st.success("Session created successfully, please click on application from left menu.")