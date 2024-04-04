import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatGooglePalm
from htmlTemplates import css
from PIL import Image
import traceback
from utils.langchain_utils import store_documents_in_database


def get_conversation_chain():
    # for RetrievalQA
    print(st.session_state.keys())
    if "conversation_chain" not in st.session_state:
        llm = ChatGooglePalm(temprature = 0.5, model_kwargs={"max_length": 200})
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

    if output:
        st.session_state.chat_history=output["chat_history"]
        
        print(st.session_state.chat_history)
        for i, message in enumerate(st.session_state.chat_history):
            if i%2 == 0:
                # st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
                st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message.content)
            else:
                # st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
                st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(message.content)
                # st.chat_message("user", avatar="🤖").write(message.content)
    else:
        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(query)
        st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write("I don't know the answer.")


st.set_page_config(page_title="Q and A using Documents", page_icon=":book:")
st.write(css, unsafe_allow_html= True)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "chat_history" not in st.session_state: 
    st.session_state["chat_history"] = []
if "disabled" not in st.session_state:
    st.session_state.disabled = True
def enabled():
    st.session_state.disabled = False

st.title("🤖 Q&A using documents")
query = st.chat_input("Enter question here:", disabled=st.session_state.disabled)

with st.sidebar:
    st.title("Upload your documents..")
    documents = st.file_uploader(label="Choose a file", accept_multiple_files=True)
    button = st.button(label="Process", on_click = enabled)
    if button:
        if "chat_history" in st.session_state: st.session_state["chat_history"] = []
        if "conversation_chain" in st.session_state: del st.session_state["conversation_chain"]
        if "vector_store" in st.session_state: del st.session_state["vector_store"]
        with st.spinner('Processing...'):
            if documents:
                print(documents)
                st.session_state.vector_store = store_documents_in_database(documents)
                st.success('Document processed!')



if query:
    with st.spinner('Thinking...'):
        handle_user_questions(query)