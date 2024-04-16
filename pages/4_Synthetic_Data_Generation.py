import streamlit as st
from utils.langchain_utils import process_documents


st.set_page_config(page_title="Synthetic Data Generation", page_icon=":book:", layout="wide")
st.title("🤖 Synthetic Data Generation")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')



num_rows = st.sidebar.slider("Selet output data size", min_value=1000, max_value=200000, value=5000, step=1000)
documents = st.file_uploader(label="Upload your sample data for model to learn.", type="csv")
sample_data = open("./sample_data/synthetic_data_sample.csv").read()
st.download_button('Download sample file', sample_data, "synthetic_data_sample.csv", "text/csv")

if documents:
    with st.spinner('Processing...'):
        print(documents)
        data = process_documents(documents, num_rows)
        csv = convert_df(data)
        st.write("Output Preview.")
        st.dataframe(data.head())
        st.write("Click below to download entire output file.")
        st.download_button(
                            "Download Result",
                            csv,
                            "synthetic_data.csv",
                            "text/csv"
                        )