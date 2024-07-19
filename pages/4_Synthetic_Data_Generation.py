import streamlit as st
from utils.langchain_utils import process_documents
import streamlit as st
import pandas as pd
import json
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.multi_table import HMASynthesizer
from sdv.metadata import SingleTableMetadata, MultiTableMetadata
from sdv.utils import drop_unknown_references
import traceback

st.set_page_config(page_title="Synthetic Data Generation", page_icon=":book:", layout="wide")
st.title("🤖 Synthetic Data Generation")

@st.cache_data
def process_documents_single_table(documents, metadata_file, scale):
    if documents:
        try:
            df = pd.read_csv(documents)
            if metadata_file:
                metadata_json = json.load(metadata_file)
                temp_metadata_file = 'temp_metadata.json'
                with open(temp_metadata_file, 'w') as f:
                    json.dump(metadata_json, f)
                
                metadata = SingleTableMetadata.load_from_json(temp_metadata_file)
                synthesizer = GaussianCopulaSynthesizer(metadata)
                synthesizer.fit(df)
                synthetic_data = synthesizer.sample(num_rows=(scale * df.shape[0]))
                return synthetic_data
            else:
                st.error("Please upload a metadata JSON file.")
                return None
        except Exception as e:
            st.error(f"Error loading metadata or generating synthetic data: {e}")
            return None
    else:
        st.error("Please upload a CSV file.")
        return None

@st.cache_data
def process_documents_multi_table(documents_list, metadata_file, scale):
    if all(documents_list) and metadata_file:
        try:
            # Load metadata from JSON file
            metadata_json = json.load(metadata_file)
            temp_metadata_file = 'temp_metadata.json'
            with open(temp_metadata_file, 'w') as f:
                json.dump(metadata_json, f)
            
            # Load metadata using MultiTableMetadata
            metadata = MultiTableMetadata.load_from_json(temp_metadata_file)
            
            # print(f"metadata:{metadata}")
            
            # Initialize a dictionary to hold dataframes for each table
            dataframes = {}
            for doc in documents_list:
                table_name = doc.name.split('.')[0]  # Use file name as table name
                df = pd.read_csv(doc)
                dataframes[table_name] = df               
                
            df_cleaned = drop_unknown_references(dataframes, metadata)
            synthesizer = HMASynthesizer(metadata)
            synthesizer.fit(df_cleaned)
            synthetic_data = synthesizer.sample(scale)   
            # print(synthetic_data)      
            return synthetic_data
        except Exception as e:
            st.error(f"Error loading metadata or generating synthetic data: {e}")
            print(traceback.format_exc())
            return None
    else:
        st.error("Please upload all CSV files and metadata JSON file.")
        return None

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

sample_data = open("./sample_data/Securities.csv").read()
sample_rule = open("./sample_data/Securities_rule.json").read()
st.sidebar.download_button('Download sample data', sample_data, "Securities.csv", "text/csv")
st.sidebar.download_button('Download sample rule', sample_rule, "Securities_rule.json", "text/csv")

scale = st.sidebar.slider("Selet input/output scale size", min_value=1, max_value=100, value=5, step=5)
metadata_file = st.sidebar.file_uploader(label="Upload metadata JSON file", type="json")

documents = st.file_uploader(label="Upload your sample data for the model to learn.", type="csv", accept_multiple_files=True)

if documents and metadata_file:
    button = st.button("Generate Data")
    if button:
        with st.spinner('Processing...'):
            number_of_documents = len(documents)
            if number_of_documents == 1:            
                data = process_documents_single_table(documents[0], metadata_file, scale)
                if data is not None:
                    csv = convert_df(data)
                    st.write("Output Preview.")
                    st.dataframe(data.head(20))
                    st.write("Click below to download the entire output file.")
                    st.download_button(
                        "Download Result",
                        csv,
                        "synthetic_data.csv",
                        "text/csv"
                    )

            elif number_of_documents > 1:            
                multi_table_data = process_documents_multi_table(documents, metadata_file, scale)
                for table, df in multi_table_data.items():
                    if df is not None:
                        csv = convert_df(df)
                        st.write("Output Preview.")
                        st.dataframe(df.head(5))
                        st.write("Click below to download the entire output file.")
                        st.download_button(
                            f"Download {table}",
                            csv,
                            f"{table}.csv",
                            "text/csv"
                        )
else:
    st.info("Please upload files and select the data type to generate synthetic data.")        