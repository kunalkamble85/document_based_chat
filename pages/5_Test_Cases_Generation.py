import streamlit as st
from utils.langchain_utils import get_test_cases, generate_tests_using_google


st.title("🤖 Test Cases Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


option = st.selectbox("How would you like to generate test cases?", ("", "Code", "Business Use Case"))
documents = st.file_uploader(label="Choose a BRD file", type=["doc","txt"])
user_input = st.text_area("or Insert your code or use Case", height= 200)
button = st.button(label="Generate Tests")
if button:
   with st.spinner('Processing...'):
      if option != "" and (user_input !="" or documents):
         if documents:
            user_input = documents.read()
         # output = get_test_cases(user_input, option)
         output = generate_tests_using_google(user_input, option)
         st.write("Click below button to download test cases generated.")
         st.download_button('Download test cases', output, "test_cases.txt")