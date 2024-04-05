# from dotenv import load_dotenv
# load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
st.title("🤖 Welcome to LLM Home.")

userid = st.text_input("Enter your user id.")
delete_history = st.checkbox("Delete historical documents")

button = st.button(label="Proceed")
if button:
    if delete_history: st.session_state.delete_history = True
    elif "delete_history" in st.session_state: del st.session_state["delete_history"]
    st.session_state.CHROMA_DB_PATH = f"./chroma_vector_database/{userid}"
    print(st.session_state.CHROMA_DB_PATH)
    st.session_state.KUZU_DB_PATH = f"./kuzu_database/{userid}"
    print(st.session_state.KUZU_DB_PATH)
    st.success("Session created successfully, please click on application from left menu.")