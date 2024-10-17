import streamlit as st
from utils.langchain_utils import generate_user_stories, display_sidebar
from utils.jira_utils import *

st.set_page_config(page_title="User Story Generation", page_icon=":book:", layout="wide")
display_sidebar()
st.title("🤖 User Story Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


how_select = st.radio("How would you like to provide input for User Story Generation?", ["Piece of input", "Upload File"], horizontal=True)

documents =None
user_input =None
if how_select == "Piece of input":
   user_input = st.text_area("Insert your code or use Case", value="Generate user stories to read csv file as input, tranform data and save into Oracle database", height= 200)
else:
   documents = st.file_uploader(label="Choose a BRD file", type=["doc","txt"])

button = st.button(label="Generate User Stories")
if button:
   with st.spinner('Processing...'):
         if documents:
            user_input = documents.read()
         if user_input:
            output = generate_user_stories(user_input)
            stories_text = create_jira_stories(output)
            st.write("Click below button to download user stories generated.")
            st.download_button('Download User Stories', stories_text, "user_stories.html")
            # st.code(output, language="python", line_numbers=False)    
            st.markdown(f"""<p style="background-color: #DEDDDD">{stories_text}</p>""", unsafe_allow_html=True)