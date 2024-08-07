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

def detect_single_table_metadata(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata

def detect_multi_table_metadata(dataframes):
    metadata = MultiTableMetadata()
    metadata.detect_from_dataframes(dataframes)
    return metadata

def update_single_table_metadata_from_json(metadata, json_file):
    try:
        json_data = json.load(json_file)
        
        # Update columns metadata
        for column_name, column_metadata in json_data.get('columns', {}).items():
            if column_name in metadata.columns:
                metadata.columns[column_name] = column_metadata
            else:
                # Add new columns if they do not exist in the metadata
                metadata.columns[column_name] = column_metadata

        # Update primary key
        if 'primary_key' in json_data:
            metadata.primary_key = json_data['primary_key']

        # Add constraints if specified
        if 'constraint_class' in json_data and 'constraint_parameters' in json_data:
            if not hasattr(metadata, 'constraint_class') or not hasattr(metadata, 'constraint_parameters'):
                metadata.constraint_class = json_data['constraint_class']
                metadata.constraint_parameters = json_data['constraint_parameters']
        
    except Exception as e:
        st.error(f"Error updating single table metadata from JSON file: {e}")

def update_multi_table_metadata_from_json(metadata, json_file):
    try:
        json_data = json.load(json_file)

        # Update tables metadata
        for table_name, table_data in json_data.get('tables', {}).items():
            if table_name in metadata.tables:
                table_metadata = metadata.tables[table_name]
                
                # Update columns metadata
                if 'columns' in table_data:
                    for column_name, column_metadata in table_data['columns'].items():
                        if column_name in table_metadata.columns:
                            table_metadata.columns[column_name] = column_metadata
                        else:
                            # Add new columns if they do not exist in the metadata
                            table_metadata.columns[column_name] = column_metadata

                # Update primary key
                if 'primary_key' in table_data:
                    table_metadata.primary_key = table_data['primary_key']

                # Update foreign keys
                if 'foreign_keys' in table_data:
                    table_metadata.foreign_keys = table_data['foreign_keys']

                # Update relationships
                if 'relationships' in table_data:
                    table_metadata.relationships = table_data['relationships']

        # Add constraints if specified
        if 'constraint_class' in json_data and 'table_name' in json_data and 'constraint_parameters' in json_data:
            table_name = json_data['table_name']
            if table_name in metadata.tables:
                table_metadata = metadata.tables[table_name]
                if not hasattr(table_metadata, 'constraint_class') or not hasattr(table_metadata, 'constraint_parameters'):
                    table_metadata.constraint_class = json_data['constraint_class']
                    table_metadata.constraint_parameters = json_data['constraint_parameters']
                    
    except Exception as e:
        st.error(f"Error updating multi-table metadata from JSON file: {e}")

def validate_metadata(metadata, dataframes):
    try:
        if isinstance(metadata, SingleTableMetadata):
            for column_name, column_metadata in metadata.columns.items():
                if column_name not in dataframes.columns:
                    st.warning(f"Column '{column_name}' in metadata does not exist in the data.")
                    return False
        
        elif isinstance(metadata, MultiTableMetadata):
            for table_name, table_metadata in metadata.tables.items():
                if table_name in dataframes:
                    for column_name, column_metadata in table_metadata.columns.items():
                        if column_name not in dataframes[table_name].columns:
                            st.warning(f"Column '{column_name}' in metadata for table '{table_name}' does not exist in the data.")
                            return False
        return True
    except Exception as e:
        st.error(f"Error validating metadata: {e}")
        return False

def metadata_to_json(metadata):
    if isinstance(metadata, SingleTableMetadata):
        # Include constraints in single table metadata JSON
        metadata_dict = metadata.to_dict()
        if hasattr(metadata, 'constraint_class') and hasattr(metadata, 'constraint_parameters'):
            metadata_dict['constraint_class'] = metadata.constraint_class
            metadata_dict['constraint_parameters'] = metadata.constraint_parameters
        return json.dumps(metadata_dict, indent=4)
    
    elif isinstance(metadata, MultiTableMetadata):
        # Include constraints in multi-table metadata JSON
        metadata_dict = metadata.to_dict()
        for table_name, table_metadata in metadata_dict['tables'].items():
            if 'constraint_class' in table_metadata and 'constraint_parameters' in table_metadata:
                table_metadata['constraint_class'] = table_metadata['constraint_class']
                table_metadata['constraint_parameters'] = table_metadata['constraint_parameters']
        return json.dumps(metadata_dict, indent=4)

@st.cache_data
def process_single_table(df, scale, metadata_file=None):
    try:
        # Detect metadata
        metadata = detect_single_table_metadata(df)
        
        if metadata_file:
            # Update metadata with values from the uploaded JSON file
            update_single_table_metadata_from_json(metadata, metadata_file)
        
        # Validate metadata
        if not validate_metadata(metadata, df):
            return None, None
        
        # Initialize synthesizer
        synthesizer = GaussianCopulaSynthesizer(metadata)
        
        # Apply constraints if specified
        if hasattr(metadata, 'constraint_class') and hasattr(metadata, 'constraint_parameters'):
            my_constraint = {
                'constraint_class': metadata.constraint_class,
                'constraint_parameters': metadata.constraint_parameters
            }
            synthesizer.add_constraints(constraints=[my_constraint])
        
        # Fit and generate synthetic data
        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_rows=(scale * df.shape[0]))

        return metadata, synthetic_data
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return None, None

@st.cache_data
def process_multi_table(dataframes, scale, metadata_file=None):
    try:
        # Detect metadata
        metadata = detect_multi_table_metadata(dataframes)
        
        if metadata_file:
            # Update metadata with values from the uploaded JSON file
            update_multi_table_metadata_from_json(metadata, metadata_file)
        
        # Validate metadata
        if not validate_metadata(metadata, dataframes):
            return None, None
        
        # Clean dataframes
        df_cleaned = drop_unknown_references(dataframes, metadata)
        
        # Initialize synthesizer
        synthesizer = HMASynthesizer(metadata)
        
        # Apply constraints if specified
        constraints = []
        for table_name, table_metadata in metadata.tables.items():
            if hasattr(table_metadata, 'constraint_class') and hasattr(table_metadata, 'constraint_parameters'):
                my_constraint = {
                    'constraint_class': table_metadata.constraint_class,
                    'table_name': table_name,
                    'constraint_parameters': table_metadata.constraint_parameters
                }
                constraints.append(my_constraint)
        
        if constraints:
            synthesizer.add_constraints(constraints=constraints)
            
        synthesizer.fit(df_cleaned)
        # Fit and generate synthetic data
        synthetic_data = synthesizer.sample(scale)
        
        # Convert synthetic data into a dictionary for download
        dataframes_out = {table_name: synthetic_data[table_name] for table_name in metadata.tables.keys()}
        
        return metadata, dataframes_out
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        print(traceback.format_exc())
        return None, None

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

sample_data = open("./sample_data/Securities.csv").read()
sample_rule = open("./sample_data/Securities_rule.json").read()
st.sidebar.download_button('Download sample data', sample_data, "Securities.csv", "text/csv")
st.sidebar.download_button('Download sample rule', sample_rule, "Securities_rule.json", "text/csv")

scale = st.sidebar.slider("Selet input/output scale size", min_value=1, max_value=100, value=5, step=5)
metadata_file = st.sidebar.file_uploader(label="Upload metadata JSON file", type="json")

documents = st.file_uploader(label="Upload your sample data for the model to learn.", type="csv", accept_multiple_files=True)

if documents:
    button = st.button("Generate Data")
    if button:
        number_of_documents = len(documents)
        if number_of_documents == 1:
            with st.spinner('Processing...'):
                df = pd.read_csv(documents[0])
                metadata, data = process_single_table(df, scale, metadata_file)
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
                    # Provide download for metadata
                    if metadata:
                        metadata_json = metadata_to_json(metadata)
                        st.sidebar.download_button(
                            "Download Metadata",
                            metadata_json,
                            "metadata.json",
                            "application/json"
                        )

        elif number_of_documents > 1:
            with st.spinner('Processing...'):
                dataframes = {}
                for doc in documents:
                    table_name = doc.name.split('.')[0]  # Use file name as table name
                    df = pd.read_csv(doc)
                    dataframes[table_name] = df
                
                metadata, multi_table_data = process_multi_table(dataframes, scale, metadata_file)
                if multi_table_data:
                    for table, df in multi_table_data.items():
                        if df is not None:
                            csv = convert_df(df)
                            st.write(f"Output Preview for {table}.")
                            st.dataframe(df.head(5))
                            st.write("Click below to download the entire output file.")
                            st.download_button(
                                f"Download {table}",
                                csv,
                                f"{table}.csv",
                                "text/csv"
                            )
                    # Provide download for metadata
                    if metadata:
                        metadata_json = metadata_to_json(metadata)
                        st.sidebar.download_button(
                            "Download Metadata",
                            metadata_json,
                            "metadata.json",
                            "application/json"
                        )
else:
    st.info("Please upload files to generate synthetic data.")