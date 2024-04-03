import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.chat_models import ChatGooglePalm
from htmlTemplates import css
from PIL import Image
from dotenv import load_dotenv
from langchain_core.prompts import (PromptTemplate)
from utils.langchain_utils import store_documents_in_database


CHROMA_DB_PATH = "./chroma_vector_database"

load_dotenv()


# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')