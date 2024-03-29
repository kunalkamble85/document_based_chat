import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.chat_models import ChatGooglePalm
from htmlTemplates import css
from PIL import Image
# from dotenv import load_dotenv
from langchain_core.prompts import (PromptTemplate)
import traceback
import tempfile
import os

CHROMA_DB_PATH = "./chroma_vector_database"

# load_dotenv()


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



def get_prompt_template():
    template = """
        You are helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        Use the given Context and History enclosed with backticks to answer the question at the end.
        Don't try to make up an answer, if you don't know, just say that you don't know.

        Chat History:
        {chat_history}

        Context: 
        {context}

        Follow Up Input: {question}
        Standalone question or instruction:
        """
    return PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])



def get_conversation_chain(vector_store):
    # for RetrievalQA
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

    return ConversationalRetrievalChain.from_llm(llm=llm, 
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                                                 memory = memory, 
                                                 chain_type = "stuff",
                                                 verbose= True, 
                                                 return_source_documents=True)
                                                #  combine_docs_chain_kwargs={"prompt": get_prompt_template()})



def store_documents_in_database(docs):
    text = []
    for file in docs:
        try:
            file_extension = os.path.splitext(file.name)[1]
            print(file_extension)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            elif file_extension == ".doc" or file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".xls" or file_extension == ".xlsx":
                loader = UnstructuredExcelLoader(temp_file_path)
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
            print(f"Saved file to vector database: {file}")
        except:
            print(traceback.format_exc())
            print(f"Error while loading file: {file}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n"], length_function=len)
        chunks = text_splitter.split_documents(text)
        # to use own embedding
        # embeddings = HuggingFaceEmbeddings(model_name='google/flan-t5-xxl',model_kwargs={'device': 'cpu'})
        # db = st.session_state.vectordb.from_texts(texts = chunks)
        vector_store.from_documents(documents = chunks, embedding=embedding_function)


def handle_user_questions(query):
    # for RetrievalQA
    # output = st.session_state.conversation_chain({"query": query})
    # for ConversationalRetrievalChain
    try:
        output = st.session_state.conversation_chain.invoke({"question": query})
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


with st.spinner('Initializing database...'):
    vector_store = Chroma(embedding_function = embedding_function)
    # vector_retriever = st.session_state.vectordb.as_retriever()

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = get_conversation_chain(vector_store)
if "chat_history" in st.session_state:
    st.session_state.chat_history=None
if "disabled" not in st.session_state:
    st.session_state.disabled = True

def enabled():
    st.session_state.disabled = False

    
st.title("Q&A using documents")
query = st.chat_input("Enter question here:", disabled=st.session_state.disabled)

with st.sidebar:
    st.title("Upload your documents..")
    documents = st.file_uploader(label="Choose a file", accept_multiple_files=True)
    button = st.button(label="Process", on_click = enabled)
    if button:
        with st.spinner('Processing...'):
            if documents:
                print(documents)
                store_documents_in_database(documents)
                st.success('Document processed!')
                # st.session_state.chat_history=None


if query:
    handle_user_questions(query)