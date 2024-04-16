import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatGooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css
from PIL import Image
import traceback
from utils.langchain_utils import store_documents_in_database
from utils.langchain_utils import clear_vectordb
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

st.set_page_config(page_title="Q and A using Documents", page_icon=":book:", layout="wide")

def get_conversation_chain():
    # for RetrievalQA
    print(st.session_state.keys())
    if "conversation_chain" not in st.session_state:
        llm = ChatGooglePalm(temprature = 0.5, model_kwargs={"max_length": 200})
        # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temprature = 0.1)
        # llm = GoogleGenerativeAI(model="gemini-pro", temprature=0.5, model_kwargs={"max_length": 200})

        # memory = ConversationBufferMemory(memory_key="chat_history", output_key='result', return_messages = True,
        #                                   return_source_documents=True)
        # return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(),
        #                             memory = memory, verbose= True, return_source_documents=True, )

        # for ConversationalRetrievalChain
        memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True,
                                        return_source_documents=True)
        # llm = load_hugging_face_llm()

        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                    retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                                                    memory = memory, 
                                                    chain_type = "stuff",
                                                    verbose= True, 
                                                    return_source_documents=True)
                                                    #  combine_docs_chain_kwargs={"prompt": get_prompt_template()})
        st.session_state.conversation_chain = qa_chain
    return st.session_state.conversation_chain


def handle_user_questions(query):
    # for RetrievalQA
    # output = st.session_state.conversation_chain({"query": query})
    # for ConversationalRetrievalChain
    try:
        output = get_conversation_chain().invoke({"question": query})
    except:
        print(traceback.format_exc())
        output = None
        print("Error while getting response")

    if "chat_history" in st.session_state or output:
        if output:
            st.session_state.chat_history=output["chat_history"]
        # print(st.session_state.chat_history)
        for i, message in enumerate(st.session_state.chat_history):
            # print(type(message))
            if i%2 == 0:
                # st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
                st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message.content)
            else:
                # st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
                st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(message.content)
                # st.chat_message("user", avatar="🤖").write(message.content)

    if not output:
        st.session_state.chat_history.append(HumanMessage(content = query))
        st.session_state.chat_history.append(AIMessage(content = "I don't know the answer."))
        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(query)
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write("I don't know the answer.")


if "CHROMA_DB_PATH" not in st.session_state:
    st.error("Please enter your user id in home page.")
else:
    st.write(css, unsafe_allow_html= True)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if "chat_history" not in st.session_state: 
        st.session_state["chat_history"] = []
    if "disabled" not in st.session_state:
        st.session_state.disabled = True
    if "vector_store" not in st.session_state:
        with st.spinner('Initializing database...'):
            st.session_state.vector_store = Chroma(persist_directory=st.session_state.CHROMA_DB_PATH, embedding_function=embedding_function)
    if "delete_history" in st.session_state:
        clear_vectordb(embedding_function)
        
    def enabled():
        st.session_state.disabled = False

    st.title("🤖 Q&A using documents")
    query = st.chat_input("Enter question here:")

    with st.sidebar:
        st.title("Upload your documents..")
        documents = st.file_uploader(label="Choose a file", accept_multiple_files=True)
        button = st.button(label="Process")
        if button:
            with st.spinner('Processing...'):
                if documents:
                    print(documents)
                    store_documents_in_database(documents)
                    st.success('Document processed!')


    if query:
        with st.spinner('Thinking...'):
            handle_user_questions(query)