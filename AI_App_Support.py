from dotenv import load_dotenv
import streamlit as st
from utils.ui_utils import display_sidebar

load_dotenv()

st.set_page_config(page_title="🤖 AI App Suppport", page_icon=":book:", layout="wide")
st.title("🤖 AI Application Support")
display_sidebar()

task = st.radio("Select issue source", ["Email Text", "Jira ID"], horizontal=True)
if task == "Email Text":
    input_text = st.text_area(label="Email content", value="", height= 200)
else:
    input_text = st.text_input("Jira ID")
    
button = st.button(label="Proceed")

if button:
    if input_text and task == "Jira ID":
        print(f"User has selected Jira:{input_text}")
        if not input_text.isdigit():
            st.error("Please enter valid Jira ID")
    elif input_text:
        print(f"User has selected Email:{input_text}")
    else:
        st.error("Please enter valid input")
