import streamlit as st
from utils.langchain_utils import process_documents


st.title("🤖 Synthetic Data Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')



num_rows = st.sidebar.slider("Output size", min_value=1000, max_value=200000, value=5000, step=1000)
documents = st.file_uploader(label="Upload your sample data", type="csv")
if documents:
    with st.spinner('Processing...'):
        print(documents)
        data = process_documents(documents, num_rows)
        csv = convert_df(data)
        st.download_button(
                            "Download",
                            csv,
                            "file.csv",
                            "text/csv",
                            key='download-csv'
                        )