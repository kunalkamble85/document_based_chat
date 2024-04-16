import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatGooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css
from PIL import Image
import traceback

st.set_page_config(page_title="Q and A", page_icon=":book:", layout="wide")

def get_conversation_chain():
    # for RetrievalQA
    print(st.session_state.keys())
    if "QA_conversation_chain" not in st.session_state:
        llm = ChatGooglePalm(temprature = 0.5, model_kwargs={"max_length": 200})
        # llm = ChatGoogleGenerativeAI(model="gemini-pro", temprature = 0.1)
        # llm = GoogleGenerativeAI(model="gemini-pro", temprature=0.5, model_kwargs={"max_length": 200})

        # memory = ConversationBufferMemory(memory_key="history", output_key='result', return_messages = True,
        #                                   return_source_documents=True)
        # return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(),
        #                             memory = memory, verbose= True, return_source_documents=True, )

        # for ConversationalRetrievalChain
        memory = ConversationBufferMemory()
        # llm = load_hugging_face_llm()

        qa_chain = ConversationChain(llm=llm, memory = memory)
        st.session_state.QA_conversation_chain = qa_chain
    return st.session_state.QA_conversation_chain


def handle_user_questions(query):
    # for RetrievalQA
    # output = st.session_state.conversation_chain({"query": query})
    # for ConversationalRetrievalChain
    try:
        output = get_conversation_chain().invoke(query)
        print(output)
    except:
        print(traceback.format_exc())
        output = None
        print("Error while getting response")

    if output:
        if "history" in output:
            history = output["history"].split("\n")
            if output["history"] !="":
                for i, message in enumerate(history):
                    if i%2 == 0:
                        # st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
                        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message)
                    else:
                        # st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
                        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(message)
                        # st.chat_message("user", avatar="🤖").write(message.content)
        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(output["input"])
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(output["response"])
    else:
        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(query)
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write("I don't know the answer.")

st.write(css, unsafe_allow_html= True)

# if "history" not in st.session_state: 
#     st.session_state["history"] = []

st.title("🤖 Chat Bot")
query = st.chat_input("Enter question here:")

if query:
    with st.spinner('Thinking...'):
        handle_user_questions(query)