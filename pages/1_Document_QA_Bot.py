import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatGooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from utils.file_reader import *
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores.faiss import DistanceStrategy
import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Q and A using Documents", page_icon=":book:", layout="wide")
with st.container():
    if "CHROMA_DB_PATH" not in st.session_state:
        st.error("Please enter your user id in home page.")
        st.stop()

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs = {"normalize_embeddings": True})

embeddings = st.session_state.embeddings

st.title("🤖 Q&A using documents")

def get_conversation_chain():
    prompt_template = """You are Document Q&A or Summarizer buddy. Give answer or summary to the questions solely from only provided text also your response should be in bullets or points and language must be English only.
    if you can't find the answer in the text you must say you don't know. You must ignore any information that is not relevent to the question. Do not give any general answers apart from text provided.
    Give short answers for the questions.
    question:\{question}\nText:{context}
    """
    model = ChatGooglePalm(temprature = 0.1, model_kwargs={"max_length": 200})
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temprature = 0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff",prompt=prompt)
    return chain
# print(os.name)
if os.name == "nt":
    DATA_DIR = "C:\\kunal\\work\\repo\\document_based_chat\\vector_database\\kunalkamble85"
else:
    DATA_DIR = st.session_state.CHROMA_DB_PATH

os.makedirs(DATA_DIR, exist_ok=True)

def is_file_processed(filename):
    return os.path.exists(os.path.join(DATA_DIR, filename))

def search_text_fs(question, filename):
    vectordb = FAISS.load_local(os.path.join(
        DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization = True, distance_strategy = DistanceStrategy.COSINE)
    data =[]
    vectordb.index.distance_measure = "cosine"

    if filename == ["All"]:
        results = vectordb.similarity_search_with_score(query=question, k=10)
    else:
        results = vectordb.similarity_search_with_score(query=question, k=10, filter={"filename":filename})
    print(results)
    for doc, score in results:
        if score < 0.99:
            data.append(doc)
    with st.expander("Check Source Documents"):
        if len(data)>0:
            for i in data:
                st.info(f"""{i.metadata.get('filename')}
                        {i.page_content}""")
        else:
            st.info("No Source Found")
            data = [Document(page_content="No Source", metadata={"filename":"None"})]
    return data

def user_input(user_question, files):
    docs = search_text_fs(user_question, files)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=False)
    return response["output_text"]

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "doc_messages" not in st.session_state:
    # st.session_state["doc_messages"] = [{AIMessage(content = "Query your documents")}]
    st.session_state["doc_messages"] = [{"role":"assistant" , "content":"Query your documents"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

def check_file_exists_fs(filename_check):
    try:
        vectordb = FAISS.load_local(os.path.join(DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
        flag = False
        for i, j in enumerate(vectordb.docstore._dict.items()):
            if j[1].metadata.get("filename") == filename_check:
                flag = True
                break
        return flag
    except Exception as e:
        print(e)
        return False
    
def delete_existing_docs_fs(filename_check):
    vectordb = FAISS.load_local(os.path.join(DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    ids_to_delete =[]
    for i, j in enumerate(vectordb.docstore._dict.items()):
        if j[1].metadata.get("filename") == filename_check:
            ids_to_delete.append(j[0])
    
    if ids_to_delete:
        vectordb.delete(ids=ids_to_delete)
        vectordb.save_local(os.path.join(DATA_DIR, "faiss_index"))
    else:
        print("Nothing to delete")
    return ids_to_delete

def save_to_vector_db_fs(texts, filename, override_task, file_check_status):
    if override_task and file_check_status:
        if file_check_status:
            delete_existing_docs_fs(filename_check=filename)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n","\r\n"])
    texts = text_splitter.create_documents([texts], metadatas=[{"filename":filename}])
    st.toast(f"Number of chunks:{len(texts)}")
    try:
        if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
            vector_store = st.session_state["vector_store"]
        else:
            vectordb = FAISS.load_local(os.path.join(DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
            vector_store = FAISS.from_documents(texts, embeddings)
            vectordb.merge_from(vector_store)
            pkl = vectordb.serialize_to_bytes()
            vectordb = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl)
            vectordb.save_local(os.path.join(DATA_DIR, "faiss_index"))
    except Exception as e:
        print(e)
        print("No database available.")
        vectordb = FAISS.from_documents(texts, embeddings)
        vectordb.save_local(os.path.join(DATA_DIR, "faiss_index"))

def update_vector_store(file_name, file_content):
    override_task = False
    file_check_status = check_file_exists_fs(file_name)
    file_exists = check_file_exists_fs(file_name)
    if file_exists:
        override_task = True
    save_to_vector_db_fs(file_content, file_name, override_task, file_check_status)

def get_all_filenames_from_vector_store():
    try:
        filenames = set()
        vectordb = FAISS.load_local(os.path.join(DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
        for i, j in vectordb.docstore._dict.items():
            filename = j.metadata.get("filename")
            if filename:
                filenames.add(filename)
        return list(filenames)
    except Exception as e:
        print(e)
        return []

def display_processed_files():
    processed_files = get_all_filenames_from_vector_store()
    if processed_files:
        st.write("Processed Files:", processed_files)
    else:
        st.write("No files processed yet.")

task = st.radio("task", options=["Upload File", "Document Q&A", "Delete File"], horizontal=True, label_visibility="hidden", index=1)

clear_history = st.button("Clear conversation history")
if clear_history:
    st.session_state.conversation = None
    # st.session_state["doc_messages"] = [{AIMessage(content = "Query your documents")}]
    st.session_state["doc_messages"] = [{"role":"assistant" , "content":"Query your documents"}]
    st.session_state["chat_history"] = []
    st.session_state["vector_store"] = (None)

if task == "Upload File":
    user_uploads = st.file_uploader("Upload your documents", accept_multiple_files=True)
    if user_uploads is not None:
        if st.button("Upload"):
            with st.spinner("Processing documents"):
                for uploaded_file in user_uploads:
                    file_name = uploaded_file.name
                    if uploaded_file.type == "application/pdf":
                        raw_text = get_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text = get_text_from_docx(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        raw_text = get_text_from_txt(uploaded_file)
                    elif uploaded_file.type == "text/csv":
                        raw_text = get_text_from_csv(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        raw_text = get_text_from_xlsx(uploaded_file)
                    
                    update_vector_store(file_name, raw_text)
                    st.session_state.documents_processed = True
                    st.info(f"File {file_name} uploaded successfully.")
if task == "Delete File":
    fileslist = get_all_filenames_from_vector_store()
    if len(fileslist):
        default = fileslist[0]
    else:
        default = None
    files = st.multiselect("Choose files",options=fileslist, label_visibility="hidden", placeholder="Choose file for Q&A", default=default)
    if files:
        st.write("File choosen to delete:")
        st.markdown(files)
        if st.button("Delete Files"):
            with st.spinner("Deleting file.."):
                for i in files:
                    delete_existing_docs_fs(i)
                    st.info(f"File {i} is deleted successfully.")
else:
    fileslist = get_all_filenames_from_vector_store()
    if len(fileslist):
        if len(fileslist) > 1:
            default = "All"
            fileslist = ["All"] + fileslist
        else:
            default = fileslist[0]
    else:
        default = None
    files = st.multiselect("Choose File for Q&A", options=fileslist, label_visibility="hidden", placeholder="Choose File for Q&A", default=default)
    if files:
        for message in st.session_state["doc_messages"]:
            # with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message["content"])
            else:
                st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(message["content"])
                    # st.write(message["content"])

        if user_query := st.chat_input("Enter your question here"):
            st.session_state["doc_messages"].append({"role":"user","content":user_query})
            st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(user_query)
            # with st.chat_message("user"):
            #     st.markdown(user_query)
            with st.spinner("Thinking.."):
                st.session_state["chat_history"].append({"role":"user","content":user_query})
                response = user_input(user_query, files)
                st.session_state["chat_history"].append({"role":"assistant","content":response})
                st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(response)
                # with st.chat_message("assistant"):
                #     st.write(response)
                st.session_state["doc_messages"].append({"role":"assistant","content":response})