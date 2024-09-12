import streamlit as st
import pandas as pd
import zipfile
import json
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.multi_table import HMASynthesizer
from sdv.metadata import SingleTableMetadata, MultiTableMetadata
from sdv.utils import drop_unknown_references

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

def apply_metadata_from_json(metadata, json_data):
    try:
        if isinstance(metadata, SingleTableMetadata):
            for column_name, column_metadata in json_data.get('columns', {}).items():
                if column_name in metadata.columns:
                    metadata.columns[column_name] = column_metadata
                else:
                    metadata.columns[column_name] = column_metadata

            if 'primary_key' in json_data:
                metadata.primary_key = json_data['primary_key']

            if 'constraint_class' in json_data and 'constraint_parameters' in json_data:
                metadata.constraint_class = json_data['constraint_class']
                metadata.constraint_parameters = json_data['constraint_parameters']

        elif isinstance(metadata, MultiTableMetadata):
            for table_name, table_data in json_data.get('tables', {}).items():
                if table_name in metadata.tables:
                    table_metadata = metadata.tables[table_name]

                    if 'columns' in table_data:
                        for column_name, column_metadata in table_data['columns'].items():
                            if column_name in table_metadata.columns:
                                table_metadata.columns[column_name] = column_metadata
                            else:
                                table_metadata.columns[column_name] = column_metadata

                    if 'primary_key' in table_data:
                        table_metadata.primary_key = table_data['primary_key']

                    if 'foreign_keys' in table_data:
                        table_metadata.foreign_keys = table_data['foreign_keys']

                    if 'relationships' in table_data:
                        table_metadata.relationships = table_data['relationships']

            if 'constraint_class' in json_data and 'table_name' in json_data and 'constraint_parameters' in json_data:
                table_name = json_data['table_name']
                if table_name in metadata.tables:
                    table_metadata = metadata.tables[table_name]
                    table_metadata.constraint_class = json_data['constraint_class']
                    table_metadata.constraint_parameters = json_data['constraint_parameters']

    except Exception as e:
        st.error(f"Error applying metadata from JSON: {e}")

def update_single_table_metadata_from_json(metadata, json_file):
    try:
        json_data = json.load(json_file)
        
        for column_name, column_metadata in json_data.get('columns', {}).items():
            if column_name in metadata.columns:
                metadata.columns[column_name] = column_metadata
            else:
                metadata.columns[column_name] = column_metadata

        if 'primary_key' in json_data:
            metadata.primary_key = json_data['primary_key']

        if 'constraint_class' in json_data and 'constraint_parameters' in json_data:
            if not hasattr(metadata, 'constraint_class') or not hasattr(metadata, 'constraint_parameters'):
                metadata.constraint_class = json_data['constraint_class']
                metadata.constraint_parameters = json_data['constraint_parameters']
        
    except Exception as e:
        st.error(f"Error updating single table metadata from JSON file: {e}")

def update_multi_table_metadata_from_json(metadata, json_file):
    try:
        json_data = json.load(json_file)

        for table_name, table_data in json_data.get('tables', {}).items():
            if table_name in metadata.tables:
                table_metadata = metadata.tables[table_name]
                
                if 'columns' in table_data:
                    for column_name, column_metadata in table_data['columns'].items():
                        if column_name in table_metadata.columns:
                            table_metadata.columns[column_name] = column_metadata
                        else:
                            table_metadata.columns[column_name] = column_metadata

                if 'primary_key' in table_data:
                    table_metadata.primary_key = table_data['primary_key']

                if 'foreign_keys' in table_data:
                    table_metadata.foreign_keys = table_data['foreign_keys']

                if 'relationships' in table_data:
                    table_metadata.relationships = table_data['relationships']

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
        metadata_dict = metadata.to_dict()
        if hasattr(metadata, 'constraint_class') and hasattr(metadata, 'constraint_parameters'):
            metadata_dict['constraint_class'] = metadata.constraint_class
            metadata_dict['constraint_parameters'] = metadata.constraint_parameters
        return json.dumps(metadata_dict, indent=4)
    
    elif isinstance(metadata, MultiTableMetadata):
        metadata_dict = metadata.to_dict()
        for table_name, table_metadata in metadata_dict['tables'].items():
            if 'constraint_class' in table_metadata and 'constraint_parameters' in table_metadata:
                table_metadata['constraint_class'] = table_metadata['constraint_class']
                table_metadata['constraint_parameters'] = table_metadata['constraint_parameters']
        return json.dumps(metadata_dict, indent=4)

@st.cache_data
def process_single_table_Generic(df, scale, metadata_file=None):
    try:
        metadata = detect_single_table_metadata(df)
        
        if metadata_file:
            update_single_table_metadata_from_json(metadata, metadata_file)   
        
        if not validate_metadata(metadata, df):
            return None, None
        
        synthesizer = GaussianCopulaSynthesizer(metadata)
        
        if hasattr(metadata, 'constraint_class') and hasattr(metadata, 'constraint_parameters'):
            my_constraint = {
                'constraint_class': metadata.constraint_class,
                'constraint_parameters': metadata.constraint_parameters
            }
            synthesizer.add_constraints(constraints=[my_constraint])
        
        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_rows=(scale * df.shape[0]))

        return metadata, synthetic_data
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return None, None

@st.cache_data
def process_multi_table_Generic(dataframes, scale, metadata_file=None):
    try:
        metadata = detect_multi_table_metadata(dataframes)
        
        if metadata_file:
            update_multi_table_metadata_from_json(metadata, metadata_file)
        
        if not validate_metadata(metadata, dataframes):
            return None, None
        
        df_cleaned = drop_unknown_references(dataframes, metadata)

        synthesizer = HMASynthesizer(metadata)
        
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
        synthetic_data = synthesizer.sample(scale)
        
        dataframes_out = {table_name: synthetic_data[table_name] for table_name in metadata.tables.keys()}
        
        return metadata, dataframes_out
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return None, None
def process_single_table_Scenario(df, scale, metadata_json=None):
    try:
        metadata = detect_single_table_metadata(df)

        if metadata_json:
            json_data = json.loads(metadata_json)
            apply_metadata_from_json(metadata, json_data)

        if not validate_metadata(metadata, df):
            return None, None

        synthesizer = GaussianCopulaSynthesizer(metadata)
        
        if hasattr(metadata, 'constraint_class') and hasattr(metadata, 'constraint_parameters'):
            my_constraint = {
                'constraint_class': metadata.constraint_class,
                'constraint_parameters': metadata.constraint_parameters
            }
            synthesizer.add_constraints(constraints=[my_constraint])
        
        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_rows=(scale * df.shape[0]))

        return metadata, synthetic_data
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return None, None
    
def process_multi_table_Scenario(dataframes, scale, metadata_json=None):
    try:
        metadata = detect_multi_table_metadata(dataframes)

        if metadata_json:
            json_data = json.loads(metadata_json)
            apply_metadata_from_json(metadata, json_data)

        if not validate_metadata(metadata, dataframes):
            return None, None
        
        df_cleaned = drop_unknown_references(dataframes, metadata)

        synthesizer = HMASynthesizer(metadata)
        
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
        synthetic_data = synthesizer.sample(scale)
        
        dataframes_out = {table_name: synthetic_data[table_name] for table_name in metadata.tables.keys()}
        
        return metadata, dataframes_out
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return None, None


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Define local rules and paths for scenarios
local_rules = {
    "Securities": r"C:\SDG\rules\Securities.json",
    "Benchmark_Provider": r"C:\SDG\rules\Benchmark_Provider.json",
    "Portfolios": r"C:\SDG\rules\Portfolios.json",
    "Counter_Parties": r"C:\SDG\rules\Counter_parties.json",
    "Benchmarks": r"C:\SDG\rules\Benchmarks.json",
    "Benchmark_Constituent": r"C:\SDG\rules\Benchmark_Constituent.json",
    "Holding": r"C:\SDG\rules\Holding.json"
}

local_paths = {
    "Securities": r"C:\SDG\samplefile\Securities.csv",
    "Benchmark_Provider": r"C:\SDG\samplefile\Benchmark_Provider.csv",
    "Portfolios": r"C:\SDG\samplefile\Portfolios.csv",
    "Counter_Parties": r"C:\SDG\samplefile\Counter_Parties.csv",
    "Benchmarks": [r"C:\SDG\samplefile\Benchmarks.csv",
                   r"C:\SDG\samplefile\Benchmark_Provider.csv"],
    "Benchmark_Constituent": [
        r"C:\SDG\samplefile\Benchmark_Constituent.csv",
        r"C:\SDG\samplefile\Benchmarks.csv",
        r"C:\SDG\samplefile\Securities.csv"
    ],
    "Holding": [
        r"C:\SDG\samplefile\Holding.csv",
        r"C:\SDG\samplefile\Benchmarks.csv",
        r"C:\SDG\samplefile\Securities.csv",
        r"C:\SDG\samplefile\Portfolios.csv"
    ]
}

sample_rules = {
    "Single Table": r"C:\SDG\rules\Securities.json",
    "Multi Table": r"C:\SDG\rules\multitablemetadata .json",
    "Partial Single Table": r"C:\SDG\rules\Singletable.json",
    "Partial Multi Table": r"C:\SDG\rules\Multitable metadata.json"
}

# Define sample files
sample_files = {
    "Single Table": r"C:\SDG\samplefile\Securities.csv",
    "Multi Table": r"C:\SDG\samplefile\samplefiles.zip"
}

mode = st.radio("Select Mode", ["Generic", "Scenario"])

if mode == "Generic":
    selected_rule = st.sidebar.selectbox("Select Sample Rule Type To Download", list(sample_rules.keys()))

    if selected_rule:
        rule_path = sample_rules[selected_rule]
        with open(rule_path, "r", encoding='utf-8') as file:
            sample_rule = file.read()
        st.sidebar.download_button(
            "Download Sample Rule",
            sample_rule,
            f"{selected_rule.replace(' ', '_')}_rule.json",
            "text/csv"
        )

    selected_file_type = st.sidebar.selectbox("Select Sample File Type To Download ", list(sample_files.keys()))

    if selected_file_type:
        file_path = sample_files[selected_file_type]
        if selected_file_type == "Multi Table":
            with open(file_path, "rb") as file:
                sample_file = file.read()
            st.sidebar.download_button(
                "Download Sample File",
                sample_file,
                f"{selected_file_type.replace(' ', '_')}_sample.zip",
                "application/zip"
            )
        else:
            with open(file_path, "r", encoding='utf-8') as file:
                sample_file = file.read()
            st.sidebar.download_button(
                "Download Sample File",
                sample_file,
                f"{selected_file_type.replace(' ', '_')}_sample.csv",
                "text/csv"
            )

    scale = st.sidebar.slider("Select input/output scale size", min_value=1, max_value=100, value=5, step=5)
    metadata_file = st.sidebar.file_uploader(label="Upload metadata JSON file", type="json")

    documents = st.file_uploader(label="Upload your sample data for the model to learn.", type="csv", accept_multiple_files=True)

    if documents:
        button = st.button("Generate Data")
        if button:
            number_of_documents = len(documents)
            if number_of_documents == 1:
                with st.spinner('Processing...'):
                    df = pd.read_csv(documents[0])
                    metadata, data = process_single_table_Generic(df, scale, metadata_file)
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
                        if doc.name.endswith('.zip'):
                            with zipfile.ZipFile(doc, 'r') as zip_file:
                                for file_name in zip_file.namelist():
                                    with zip_file.open(file_name) as file:
                                        table_name = file_name.split('.')[0]
                                        df = pd.read_csv(file)
                                        dataframes[table_name] = df
                        else:
                            table_name = doc.name.split('.')[0]
                            df = pd.read_csv(doc)
                            dataframes[table_name] = df
                    
                    metadata, multi_table_data = process_multi_table_Generic(dataframes, scale, metadata_file)
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
                        if metadata:
                            metadata_json = metadata_to_json(metadata)
                            st.sidebar.download_button(
                                "Download Metadata",
                                metadata_json,
                                "metadata.json",
                                "application/json"
                            )


elif mode == "Scenario": 
    selected_scenario = st.sidebar.selectbox("Select Scenario", [
        "Securities", "Benchmark_Provider", "Benchmarks", "Benchmark_Constituent", "Portfolios", 
        "Holding", "Counter_Parties"
    ])   

    if selected_scenario:
        scale = st.sidebar.slider("Select Scale", min_value=1, max_value=100, value=5, step=5)

        if selected_scenario in ["Benchmarks", "Benchmark_Constituent", "Holding"]:
            dataframes = {}
            for path in local_paths[selected_scenario]:
                try:
                    df = pd.read_csv(path)
                    table_name = path.split("\\")[-1].split(".")[0]
                    dataframes[table_name] = df
                except Exception as e:
                    st.error(f"Error reading file {path}: {e}")

            metadata_file_path = local_rules[selected_scenario]
            try:
                with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                    metadata_json = metadata_file.read()
                    if st.button("Generate Data"):
                        with st.spinner('Processing...'):
                            metadata, multi_table_data = process_multi_table_Scenario(dataframes, scale, metadata_json)
                            if multi_table_data:
                                for table_name, data in multi_table_data.items():
                                    st.write(f"Table: {table_name}")
                                    st.dataframe(data)
                                    csv = convert_df(data)
                                    st.download_button(
                                        label=f"Download {table_name} CSV",
                                        data=csv,
                                        file_name=f"{table_name}.csv",
                                        mime="text/csv"
                                    )
                                if metadata:
                                    metadata_json = metadata_to_json(metadata)
                                    st.sidebar.download_button(
                                        label="Download Metadata JSON",
                                        data=metadata_json,
                                        file_name="metadata.json",
                                        mime="application/json"
                                    )
            except Exception as e:
                st.error(f"Error reading metadata file {metadata_file_path}: {e}")

        else:
            try:
                df = pd.read_csv(local_paths[selected_scenario])
                metadata_file_path = local_rules[selected_scenario]
                try:
                    with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                        metadata_json = metadata_file.read()
                        if st.button("Generate Data"):
                            with st.spinner('Processing...'):
                                metadata, synthetic_data = process_single_table_Scenario(df, scale, metadata_json)
                                if synthetic_data is not None:
                                    st.dataframe(synthetic_data)
                                    csv = convert_df(synthetic_data)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="synthetic_data.csv",
                                        mime="text/csv"
                                    )
                                    if metadata:
                                        metadata_json = metadata_to_json(metadata)
                                        st.sidebar.download_button(
                                            label="Download Metadata JSON",
                                            data=metadata_json,
                                            file_name="metadata.json",
                                            mime="application/json"
                                        )
                except Exception as e:
                    st.error(f"Error reading metadata file {metadata_file_path}: {e}")
            except Exception as e:
                st.error(f"Error reading data file {local_paths[selected_scenario]}: {e}")
