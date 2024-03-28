import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from pypdf import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatGooglePalm
from htmlTemplates import css
from PIL import Image
# from dotenv import load_dotenv

CHROMA_DB_PATH = "./chroma_vector_database"

# load_dotenv()

def get_conversation_chain():
    # for RetrievalQA
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='result', return_messages = True,
                                      return_source_documents=True)
    # for ConversationalRetrievalChain
    # memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True,
    #                                   return_source_documents=True)
    # llm = load_hugging_face_llm()
    llm = ChatGooglePalm(temprature = 0.5, model_kwargs={"max_length": 200})
    # llm = GoogleGenerativeAI(model="gemini-pro", temprature=0.5)
    return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                memory = memory, verbose= True, return_source_documents=True)
    # return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
    #                             memory = memory, verbose= True, return_source_documents=True)



def store_documents_in_database(docs):
    for doc in docs:
        reader = PdfReader(doc)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n", length_function=len)
        chunks = text_splitter.split_text(text)
        # to use own embedding
        # embeddings = HuggingFaceEmbeddings(model_name='google/flan-t5-xxl',model_kwargs={'device': 'cpu'})
        # db = st.session_state.vectordb.from_texts(texts = chunks)
        vector_store.from_texts(persist_directory= CHROMA_DB_PATH, texts = chunks,
                                embedding=embedding_function)


def handle_user_questions(query):
    # for RetrievalQA
    output = st.session_state.conversation_chain({"query": query})
    # for ConversationalRetrievalChain
    # output = st.session_state.conversation_chain({"question": query})
    # print(output)
    st.session_state.chat_history=output["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            # st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
            st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message.content)
        else:
            # st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
            st.chat_message("user", avatar=Image.open('./images/ai_robot.jpg')).write(message.content)
            # st.chat_message("user", avatar="🤖").write(message.content)

st.set_page_config(page_title="Q and A using Documents", page_icon=":book:")
st.write(css, unsafe_allow_html= True)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


with st.spinner('Initializing database...'):
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH,
                          embedding_function = embedding_function)
    # vector_retriever = st.session_state.vectordb.as_retriever()

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = get_conversation_chain()
if "chat_history" in st.session_state:
    st.session_state.chat_history=None

st.title("Q&A using documents")
query = st.text_input(key="user_question" , label="Enter question here:")

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
    handle_user_questions(query)
    # st.session_state.user_question = ""

