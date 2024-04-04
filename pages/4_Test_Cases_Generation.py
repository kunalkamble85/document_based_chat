import streamlit as st
from utils.langchain_utils import get_test_cases


st.title("🤖 Test Cases Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


option = st.selectbox("How would you like to generate test cases?", ("", "Code", "Business Use Case"))
user_input = st.text_area("Insert your code or use Case")
if option != "" and user_input !="":
    output = get_test_cases(user_input, option)
    st.download_button('Download test cases', output)