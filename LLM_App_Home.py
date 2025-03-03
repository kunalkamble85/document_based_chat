# from dotenv import load_dotenv
# load_dotenv()
import streamlit as st
from utils.langchain_utils import display_sidebar
from utils.oci_utils import generate_oci_gen_ai_response

st.set_page_config(page_title="🤖 LLM Home", page_icon=":book:", layout="wide")
display_sidebar()
st.title("🤖 Welcome to Oracle Finergy LLM Demo")
# "meta.llama3.1-405b"
cloud_models = {"OCP":["meta.llama3.3-70b","meta.llama3.1-70b","meta.llama3.1-405b","cohere.command-r-plus","cohere.command-r"],"GCP":['Gemini Pro'],"AWS":['Claude Sonnet'],"OpenAI":['gpt-4o-mini']}
st.session_state.LLM_MODEL = "meta.llama3.3-70b"
st.session_state.multi_model_synth = None
userid = st.text_input("Enter your user id.")
print(f"Loging sucussful:{userid}")
# cloud_provider = st.radio(
#     "Choose your model provider",
#     ["GCP", "OCP","AWS", "OpenAI"],
#     captions = ["Google Cloud Provider", "Oracle Cloud Provider", "Amazon Web Services", "Azure Cloud Provder"], horizontal=True)
cloud_provider = "OCP"
model_name = st.radio(
    "Select OCI Generative AI Model", cloud_models.get(cloud_provider), horizontal=True)

# model_name = st.selectbox(
#     'Select LLM Model',
#     ('Google Palm','Mistral', 'Google Gemini', 'Claude Sonnet', 'llama', 'ChatGPT'))

st.markdown(''':red[All models are served by Oracle Gen AI services]''')

temprature = st.slider("Set the model temprature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
st.session_state.temprature = temprature
button = st.button(label="Proceed")
if button:
    if userid:
        st.session_state.CHROMA_DB_PATH = f"./vector_database/{userid}"
        print(st.session_state.CHROMA_DB_PATH)
        st.session_state.KUZU_DB_PATH = f"./kuzu_database/{userid}"
        print(st.session_state.KUZU_DB_PATH)
        st.session_state.LLM_MODEL = model_name
        try:
            response = generate_oci_gen_ai_response(model_name, [{"role":"user", "content":"Hello"}])
            print(f"Response from model: {response}")
            st.success("Session created successfully, please click on application from left menu.")
        except Exception as e:
            print(e)
            st.error("Error in intializing model, please try again with different model.")

    else:
        st.error("Please enter valid userid")