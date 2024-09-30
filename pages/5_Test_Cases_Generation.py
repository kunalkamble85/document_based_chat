import streamlit as st
from utils.langchain_utils import generate_tests_using_google

st.set_page_config(page_title="Test Cases Generation", page_icon=":book:", layout="wide")
st.title("🤖 Test Cases Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


option = st.radio("How would you like to generate test cases??", ["Using BRD Document", "Using Code",], horizontal=True)

how_select = st.radio("How would you like to provide input?", ["Piece of input", "Upload File"], horizontal=True)

documents =None
user_input =None
if how_select == "Piece of input":
   user_input = st.text_area("or Insert your code or use Case", value="Provide detailed test cases for login functionality for login screen",  height= 200)
else:
   documents = st.file_uploader(label="Choose a BRD file", type=["doc","txt"])

button = st.button(label="Generate Tests")
if button:
   with st.spinner('Processing...'):
         if documents:
            user_input = documents.read()
         if user_input:
            output = generate_tests_using_google(user_input, option)
            st.write("Click below button to download test cases generated.")
            st.download_button('Download test cases', output, "test_cases.txt")
            # st.code(output, language="python", line_numbers=False)        
            st.markdown(f"""<p style="background-color: #DEDDDD">{output}</p>""", unsafe_allow_html=True)