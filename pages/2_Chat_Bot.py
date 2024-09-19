import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

st.set_page_config(page_title="Chat with AI", page_icon=":book:", layout="wide")
st.title("🤖 Chat with AI")


def user_input(user_question):
    if st.session_state.LLM_MODEL == "gpt-4o-mini":
        print("Calling Open AI model")
        model = ChatOpenAI(model="gpt-4o-mini", max_tokens="200")
    else:
        print("Calling Google Palm")
        model = ChatGoogleGenerativeAI(model="gemini-pro", temprature = 0.1)
    output = model.invoke(user_question)
    return output.content

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
        response = user_input(user_query)
        st.session_state["history"].append({"role":"assistant","content":response})
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(response)
        st.session_state["chat_messages"].append({"role":"assistant","content":response})