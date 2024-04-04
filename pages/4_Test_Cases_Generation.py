import streamlit as st
from utils.langchain_utils import get_test_cases


st.title("🤖 Test Cases Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


option = st.selectbox("How would you like to generate test cases?", ("", "Code", "Business Use Case"))
documents = st.file_uploader(label="Choose a BRD file", type=["doc","txt"])
user_input = st.text_area("or Insert your code or use Case", height= 400)

if option != "" and (user_input !="" or documents):
   if documents:
      user_input = documents.read()
      output = get_test_cases(user_input, option)
   else:
      output = get_test_cases(user_input, option)
   
   st.download_button('Download test cases', output)