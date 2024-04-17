import streamlit as st
from langchain_community.chat_models import ChatGooglePalm
from htmlTemplates import css
from PIL import Image
from utils.langchain_utils import create_graph_in_database
import kuzu
from langchain.chains import KuzuQAChain
from langchain_community.graphs import KuzuGraph
import traceback

st.set_page_config(page_title="Q and A using Graph Database", page_icon=":book:", layout="wide")

def handle_user_questions(query):
    llm = ChatGooglePalm(temprature = 0.1)
    # llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temprature = 0.1)

    graph = KuzuGraph(st.session_state.kuzu_database)
    chain = KuzuQAChain.from_llm(llm, graph=graph, verbose=True)
    output = None
    try:
        output = chain.invoke(query)
    except:
        print(traceback.format_exc())
        print("error while executing graph query.")
    print(output)
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


# st.title("This page is under contruction.")
if "KUZU_DB_PATH" not in st.session_state:
    st.error("Please enter your user id in home page.")
else:
    st.write(css, unsafe_allow_html= True)

    if "kuzu_database" not in st.session_state:
        with st.spinner('Initializing graph database...'):
            st.session_state.kuzu_database = kuzu.Database(st.session_state.KUZU_DB_PATH)

    with st.sidebar:
        st.title("Upload your documents..")
        documents = st.file_uploader(label="Choose a relationship file", accept_multiple_files=False, type =["txt"])
        sample_data = open("./sample_data/graph_data_sample.txt").read()
        st.download_button('Download sample file', sample_data, "graph_data_sample.txt")
        button = st.button(label="Process")
        
        if button:
            with st.spinner('Processing...'):
                if documents:
                    print(documents)
                    create_graph_in_database(documents)
                    st.success('Document processed!')

    
    st.title("🤖 Q&A using Graph Database")
    query = st.chat_input("Enter question here:")
    if query:
        with st.spinner('Thinking...'):
            handle_user_questions(query)


