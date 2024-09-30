import streamlit as st
from utils.oci_utils import *
from PIL import Image

st.set_page_config(page_title="Chat with AI", page_icon=":book:", layout="wide")
st.title("🤖 Chat with AI")


def handle_conversation():
    conversation = st.session_state["history"]
    response = generate_oci_gen_ai_response(st.session_state.LLM_MODEL, conversation)
    return response

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [{"role":"assistant" , "content":"Ask your question"}]
if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state["chat_messages"]:
    # with st.chat_message(message["role"]):
    if message["role"] == "user":
        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message["content"])
    else:
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(message["content"])
        # st.write(message["content"])

if user_query := st.chat_input("Enter your question here"):
    st.session_state["chat_messages"].append({"role":"user","content":user_query})
    st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(user_query)
    with st.spinner("Thinking.."):
        st.session_state["history"].append({"role":"user","content":user_query})
        response = handle_conversation()
        st.session_state["history"].append({"role":"assistant","content":response})
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(response)
        st.session_state["chat_messages"].append({"role":"assistant","content":response})