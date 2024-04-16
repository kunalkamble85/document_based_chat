# from dotenv import load_dotenv
# load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
st.set_page_config(page_title="🤖 LLM Home", page_icon=":book:", layout="wide")

st.title("🤖 Welcome to LLM Home.")


userid = st.text_input("Enter your user id.")
delete_history = st.checkbox("Delete historical documents")
cloud_provider = st.radio(
    "Choose your cloud provider",
    ["GCP", "OCP","AWS", "Azure"],
    captions = ["Google Cloud Provider", "Oracle Cloud Provider", "Amazon Web Services", "Azure Cloud Provder"], horizontal=True)

model_name = st.radio(
    "Select LLM Model",
    ['Gemini Pro', "Google Palm", "Mistral","llama", "Cohore", 'Claude Sonnet', 'ChatGPT'], horizontal=True)

# model_name = st.selectbox(
#     'Select LLM Model',
#     ('Google Palm','Mistral', 'Google Gemini', 'Claude Sonnet', 'llama', 'ChatGPT'))

st.markdown(''':red[Only GCP cloud with Google Palm is avaiable right now.]''')

temprature = st.slider("Set the model temprature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

button = st.button(label="Proceed")
if button:
    if delete_history: st.session_state.delete_history = True
    elif "delete_history" in st.session_state: del st.session_state["delete_history"]
    st.session_state.CHROMA_DB_PATH = f"./chroma_vector_database/{userid}"
    print(st.session_state.CHROMA_DB_PATH)
    st.session_state.KUZU_DB_PATH = f"./kuzu_database/{userid}"
    print(st.session_state.KUZU_DB_PATH)
    st.success("Session created successfully, please click on application from left menu.")