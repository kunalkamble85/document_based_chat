import streamlit as st
from utils.file_reader import *
from utils.langchain_utils import generate_tests_using_google, display_sidebar

st.set_page_config(page_title="Test Cases Generation", page_icon=":book:", layout="wide")
display_sidebar()
st.title("🤖 Test Cases Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


option = st.radio("How would you like to generate test cases??", ["Using BRD Document", "Using Code",], horizontal=True)

how_select = st.radio("How would you like to provide input?", ["Piece of input", "Upload File"], horizontal=True)

uploaded_file =None
user_input =None
if how_select == "Piece of input":
   user_input = st.text_area("Insert your code or use Case", value="Provide detailed test cases for login functionality for login screen",  height= 200)
else:
   uploaded_file = st.file_uploader(label="Choose a BRD file")

button = st.button(label="Generate Tests")
if button:
   with st.spinner('Processing...'):
         if uploaded_file:
            try:
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
               else:
                     raw_text = get_text_from_txt(uploaded_file)                                                    
               st.info(f"File {file_name} uploaded successfully.")
               user_input = raw_text
            except Exception as e:
               st.error(f"Error while uploading File {file_name} : {e}")
               print(e)
         if user_input:
            output = generate_tests_using_google(user_input, option)
            st.write("Click below button to download test cases generated.")
            st.download_button('Download test cases', output, "test_cases.txt")
            # st.code(output, language="python", line_numbers=False)        
            st.markdown(f"""<p style="background-color: #DEDDDD">{output}</p>""", unsafe_allow_html=True)