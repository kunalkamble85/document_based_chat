import streamlit as st
import pandas as pd
import zipfile
import numpy as np
import json
import re
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.multi_table import HMASynthesizer
from sdv.metadata import SingleTableMetadata, MultiTableMetadata
from sdv.utils import drop_unknown_references
from utils.langchain_utils import generate_synthetic_data, display_sidebar

# Set up Google Gemini Pro API key and model
st.set_page_config(page_title="Synthetic Data Generation", page_icon=":book:", layout="wide")
display_sidebar()
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
            # st.write("Applying metadata JSON:", json_data)  # Debugging line
            apply_metadata_from_json(metadata, json_data)
            # st.write("Updated Metadata:", metadata.to_dict())  # Debugging line

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
            # st.write("Applying metadata JSON:", json_data)  # Debugging line
            apply_metadata_from_json(metadata, json_data)
            # st.write("Updated Metadata:", metadata.to_dict())  # Debugging line

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
    
  # Normalize the WEIGHT(%) column
def normalize_weights(df, weight_column='WEIGHT(%)'):
    total_weight = df[weight_column].sum()
    
    if total_weight == 0:
        st.error("Total weight is zero, normalization not possible.")
        return df
    
    # Normalize the weights
    df[weight_column] = df[weight_column] / total_weight
    
    # Adjust for any floating-point errors to ensure the sum is strictly 1
    difference = 1 - df[weight_column].sum()
    df[weight_column].iloc[-1] += difference  # Adjust the last value to correct any deviation
    
    return df    


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def fetch_bm_name_from_llm(num_of_records):
    try:
        generated_text = generate_synthetic_data(num_of_records)
        print(generated_text)
        benchmark_names = generated_text.split("\n")
        benchmark_names = [name.split('. ', 1)[-1].strip() for name in benchmark_names if name.strip()!="" and "20 unique" not in name]
        return benchmark_names[:num_of_records]
    except Exception as e:
        st.error(f"Error fetching BM_NAME from LLM: {e}")
        return ["Error in BM_NAME generation"]


def clean_bm_name(name):
    # Remove any characters like -, *, and strip leading/trailing whitespace
    return re.sub(r'[-*]', '', name).strip()

# Define local rules and paths for scenarios
sample_rules = {
    "Single Table": r"./sample_data/Singletable.json",
    "Multi Table": r"./sample_data/multitablemetadata .json",
    "Partial single Table": r"./sample_data/SingletableTesting.json",
    "Partial Multi Table": r"./sample_data/Multitabletesting.json"
}

# Define sample files
sample_files = {
    "Single Table": r"./sample_data/Securities.csv",
    "Multi Table": r"./sample_data/samplefiles.zip"
}

local_paths = {
    "Securities": r"./scenario_files/Securities.csv",
    "Benchmark_Provider": r"./scenario_files/Benchmark_Provider.csv",
    "Portfolios": r"./scenario_files/Portfolios.csv",
    "Counter_Parties": r"./scenario_files/counter_parties.csv",
    "Benchmarks": [r"./scenario_files/Benchmarks.csv",
                   r"./scenario_files/Benchmark_Provider.csv"],
    "Benchmark_Constituent": [
        r"./scenario_files/Benchmark_Constituent.csv",
        r"./scenario_files/Benchmarks.csv",
        r"./scenario_files/Securities.csv"

    ],
    "Holding": [
        r"./scenario_files/Holding.csv",
        r"./scenario_files/Benchmarks.csv",
        r"./scenario_files/Securities.csv",
        r"./scenario_files/Portfolios.csv"
    ],
    "Benchmark_Constituent_with_relation": [
        r"./scenario_files/Benchmark_Constituent.csv",
        r"./scenario_files/Benchmarks.csv",
        r"./scenario_files/Securities.csv"
    ],
}

local_rules = {
    "Securities": r"./scenario_rules/Securities.json",
    "Benchmark_Provider": r"./scenario_rules/Benchmark_Provider.json",
    "Portfolios": r"./scenario_rules/Portfolios.json",
    "Counter_Parties": r"./scenario_rules/Counter_parties.json",
    "Benchmarks": r"./scenario_rules/Benchmarks.json",
    "Benchmark_Constituent": r"./scenario_rules/Benchmark_Constituent.json",
    "Holding": r"./scenario_rules/Holding.json",
	"Benchmark_Constituent_with_relation": r"./scenario_rules/Benchmark_Constituent_with_relation.json"
}

mode = st.radio("Select Mode", ["Generic Data Generation", "Portfolio Management Data Generation"])

if mode == "Generic Data Generation":
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
                        st.dataframe(data.head(20), hide_index=True)
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
                                st.dataframe(df.head(5), hide_index=True)
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

if mode == "Portfolio Management Data Generation":
    selected_scenario = st.sidebar.selectbox("Select Scenario", [
        "Securities", "Benchmark_Provider", "Benchmarks", "Benchmark_Constituent", "Portfolios", 
        "Holding", "Counter_Parties","Benchmark_Constituent_with_relation"
    ])

    if selected_scenario == "Benchmark_Constituent":
       # Load Benchmarks.csv to get BM_NAME to BENCHMARK_ID mapping
        benchmarks_csv_path = local_paths["Benchmark_Constituent"][1]  # Benchmarks.csv
        benchmarks_df = pd.read_csv(benchmarks_csv_path)
        bm_name_to_id = benchmarks_df[['BM_NAME', 'BENCHMARK_ID']].dropna().drop_duplicates().set_index('BM_NAME').to_dict()['BENCHMARK_ID']
        bm_name_values = list(bm_name_to_id.keys())
        selected_bm_name = st.sidebar.selectbox("Select BM_NAME", bm_name_values)

        # Load Benchmark_Constituent.csv and Securities.csv
        benchmark_constituent_path = local_paths["Benchmark_Constituent"][0]  # Benchmark_Constituent.csv
        securities_path = local_paths["Benchmark_Constituent"][2]  # Securities.csv

        benchmark_constituent_df = pd.read_csv(benchmark_constituent_path)
        securities_df = pd.read_csv(securities_path)
        security_ids = securities_df['SECURITY_ID'].dropna().unique()


        if selected_bm_name:
            # Update BENCHMARK_ID in Benchmark_Constituent DataFrame
            benchmark_constituent_df['BENCHMARK_ID'] = bm_name_to_id.get(selected_bm_name, benchmark_constituent_df['BENCHMARK_ID'])
            
            # Text input for number of rows
            num_rows = st.sidebar.text_input("Enter the number of rows required", value="10")

            try:
                num_rows = int(num_rows)
            except ValueError:
                st.sidebar.error("Please enter a valid number.")
                num_rows = 0

            metadata_file_path = local_rules["Benchmark_Constituent"]
            try:
                with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                    metadata_json = metadata_file.read()
                    
                    # Auto-detect metadata for single-table setup
                    metadata = detect_single_table_metadata(benchmark_constituent_df)
                    
                    # Apply metadata from JSON (update columns)
                    apply_metadata_from_json(metadata, json.loads(metadata_json))
                    
                    if st.button("Generate Data"):
                        with st.spinner('Processing...'):
                            # Generate synthetic data with the number of rows equal to num_rows
                            metadata, synthetic_data = process_single_table_Scenario(benchmark_constituent_df, num_rows, metadata_json)
                            if synthetic_data is not None:
                                synthetic_data = synthetic_data.head(num_rows)  # Ensure the number of rows matches num_rows
                                
                                # Randomly assign SECURITY_ID values from the Securities table
                                synthetic_data['SECURITY_ID'] = np.random.choice(security_ids, size=num_rows, replace=True)
                                
                                # Normalize the WEIGHT(%) column
                                synthetic_data = normalize_weights(synthetic_data, weight_column='WEIGHT(%)')
                                
                                st.write(f"Synthetic Data with {num_rows} rows:")
                                st.dataframe(synthetic_data, hide_index=True)
                                
                                # Convert DataFrame to CSV
                                csv = convert_df(synthetic_data)
                                st.download_button(
                                    label="Download Synthetic Data CSV",
                                    data=csv,
                                    file_name="synthetic_data.csv",
                                    mime="text/csv"
                                )
                                
                                # Download Metadata JSON
                                if metadata:
                                    metadata_json = metadata_to_json(metadata)
                                    st.sidebar.download_button(
                                        label="Download Metadata JSON",
                                        data=metadata_json,
                                        file_name="metadata.json",
                                        mime="application/json"
                                    )
            except Exception as e:
                st.error(f"Error reading or applying metadata file {metadata_file_path}: {e}")

    elif selected_scenario == "Holding":
        # Load Portfolios.csv to get NAME to PORTFOLIO_ID mapping
        portfolios_csv_path = local_paths["Holding"][3]  # Portfolios.csv
        portfolios_df = pd.read_csv(portfolios_csv_path)
        name_to_id = portfolios_df[['NAME', 'PORTFOLIO_ID']].dropna().drop_duplicates().set_index('NAME').to_dict()['PORTFOLIO_ID']
        name_values = list(name_to_id.keys())
        selected_name = st.sidebar.selectbox("Select NAME", name_values)

        # Load Benchmarks.csv to get BM_NAME to BENCHMARK_ID mapping
        benchmarks_csv_path = local_paths["Holding"][1]  # Benchmarks.csv
        benchmarks_df = pd.read_csv(benchmarks_csv_path)
        bm_name_to_id = benchmarks_df[['BM_NAME', 'BENCHMARK_ID']].dropna().drop_duplicates().set_index('BM_NAME').to_dict()['BENCHMARK_ID']
        bm_name_values = list(bm_name_to_id.keys())
        selected_bm_name = st.sidebar.selectbox("Select BM_NAME", bm_name_values)

        # Load Holding.csv
        holding_path = local_paths["Holding"][0]  # Holding.csv
        holding_df = pd.read_csv(holding_path)

        # Load Securities.csv to get SECURITY_ID values
        securities_csv_path = local_paths["Holding"][2]  # Securities.csv
        securities_df = pd.read_csv(securities_csv_path)
        security_ids = securities_df['SECURITY_ID'].dropna().unique()

        if selected_name and selected_bm_name:
            # Update PORTFOLIO_ID in Holding DataFrame
            holding_df['PORTFOLIO_ID'] = name_to_id.get(selected_name, holding_df['PORTFOLIO_ID'])

            # Update BENCHMARK_ID in Holding DataFrame
            holding_df['BENCHMARK_ID'] = bm_name_to_id.get(selected_bm_name, holding_df['BENCHMARK_ID'])
            
            # Text input for number of rows
            num_rows = st.sidebar.text_input("Enter the number of rows required", value="10")

            try:
                num_rows = int(num_rows)
            except ValueError:
                st.sidebar.error("Please enter a valid number.")
                num_rows = 0

            metadata_file_path = local_rules["Holding"]
            try:
                with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                    metadata_json = metadata_file.read()
                    
                    # Auto-detect metadata for single-table setup
                    metadata = detect_single_table_metadata(holding_df)
                    
                    # Apply metadata from JSON (update columns)
                    apply_metadata_from_json(metadata, json.loads(metadata_json))
                    
                    if st.button("Generate Data"):
                        with st.spinner('Processing...'):
                            # Generate synthetic data with the number of rows equal to num_rows
                            metadata, synthetic_data = process_single_table_Scenario(holding_df, num_rows, metadata_json)
                            if synthetic_data is not None:
                                synthetic_data = synthetic_data.head(num_rows)  # Ensure the number of rows matches num_rows
                                
                                # Randomly assign SECURITY_ID values from the Securities table
                                synthetic_data['SECURITY_ID'] = np.random.choice(security_ids, size=num_rows, replace=True)
                                synthetic_data['MARKET_VALUE'] = synthetic_data['PAR'] * synthetic_data['CURRENT_PRICE']
                                
                                st.write(f"Synthetic Data with {num_rows} rows:")
                                st.dataframe(synthetic_data, hide_index=True)
                                
                                # Convert DataFrame to CSV
                                csv = convert_df(synthetic_data)
                                st.download_button(
                                    label="Download Synthetic Data CSV",
                                    data=csv,
                                    file_name="synthetic_data.csv",
                                    mime="text/csv"
                                )
                                
                                # Download Metadata JSON
                                if metadata:
                                    metadata_json = metadata_to_json(metadata)
                                    st.sidebar.download_button(
                                        label="Download Metadata JSON",
                                        data=metadata_json,
                                        file_name="metadata.json",
                                        mime="application/json"
                                    )
            except Exception as e:
                st.error(f"Error reading or applying metadata file {metadata_file_path}: {e}")

    elif selected_scenario == "Benchmarks":
        benchmark_provider_df = pd.read_csv(local_paths["Benchmark_Provider"])
        bm_provider_values = benchmark_provider_df['BM_PROVIDER'].unique().tolist()
        selected_bm_provider = st.sidebar.selectbox("Select Benchmark Provider", bm_provider_values)

        use_llm = st.sidebar.radio("Select LLM Option", ("WITHOUT LLM", "LLM"))

        # Only show scale slider if "WITHOUT LLM" is selected
        if use_llm == "WITHOUT LLM":
            scale = st.sidebar.slider("Select Scale", min_value=1, max_value=100, value=5, step=5, key="scale_without_llm")

        dataframes = {}
        for path in local_paths[selected_scenario]:
            try:
                df = pd.read_csv(path)
                table_name = path.split("/")[-1].split(".")[0]
                dataframes[table_name] = df
            except Exception as e:
                st.error(f"Error reading file {path}: {e}")

        benchmark_provider_df = dataframes.get("Benchmark_Provider")
        benchmarks_df = dataframes.get("Benchmarks")

        if benchmark_provider_df is not None and benchmarks_df is not None:
            bm_ids = benchmark_provider_df[benchmark_provider_df['BM_PROVIDER'] == selected_bm_provider]['BM_PROVIDER_ID'].tolist()
            benchmarks_df.loc[:, 'BM_PROVIDER_ID'] = bm_ids[0] if bm_ids else benchmarks_df['BM_PROVIDER_ID']
            dataframes["Benchmarks"] = benchmarks_df

        # Directly specify the metadata file paths based on LLM option
        if use_llm == "LLM":            
            metadata_file_path = "./scenario_rules/LLM_BM_Rule.json"
        else:  
            metadata_file_path = "./scenario_rules/Benchmarks.json"  
        try:
            with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                metadata_json = metadata_file.read()

                metadata = detect_single_table_metadata(dataframes["Benchmarks"])
                apply_metadata_from_json(metadata, json.loads(metadata_json))

                if st.button("Generate Data"):
                    with st.spinner('Processing...'):
                        if use_llm == "LLM":
                          
                            benchmarks_df['BM_NAME'] = fetch_bm_name_from_llm(len(benchmarks_df))
                            benchmarks_df = benchmarks_df.explode('BM_NAME')
                            benchmarks_df['BM_NAME'] = benchmarks_df['BM_NAME'].apply(clean_bm_name)
                            benchmarks_df['BM_NAME'].replace('', 'Unnamed Benchmark', inplace=True)

               
                            benchmarks_df['BENCHMARK_ID'] = range(1, len(benchmarks_df) + 1)

                       
                            metadata, synthetic_data = process_single_table_Scenario(benchmarks_df, 100, metadata_json)

                            if synthetic_data is not None and isinstance(synthetic_data, pd.DataFrame):
                                # Keep unique rows based on BM_NAME
                                synthetic_data_unique = synthetic_data.drop_duplicates(subset='BM_NAME')

                                st.write("Synthetic Data :")
                                st.dataframe(synthetic_data_unique, hide_index=True)

                                # Download unique rows
                                csv = convert_df(synthetic_data_unique)
                                st.download_button(
                                    label="Download Synthetic Data CSV",
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
                        else:  # Handle "WITHOUT LLM" option
                            # Process data as needed for the "WITHOUT LLM" case
                            metadata, synthetic_data_without_llm = process_single_table_Scenario(benchmarks_df, scale, metadata_json)

                            if synthetic_data_without_llm is not None and isinstance(synthetic_data_without_llm, pd.DataFrame):
                                st.write("Synthetic Data:")
                                st.dataframe(synthetic_data_without_llm.head(100), hide_index=True)

                                # Download all rows based on the scale
                                csv_without_llm = convert_df(synthetic_data_without_llm)
                                st.download_button(
                                    label="Download Synthetic Data CSV ",
                                    data=csv_without_llm,
                                    file_name="synthetic_data_without_llm.csv",
                                    mime="text/csv"
                                )
                                if metadata:
                                    metadata_json = metadata_to_json(metadata)
                                    st.sidebar.download_button(
                                        label="Download Metadata JSON ",
                                        data=metadata_json,
                                        file_name="metadata_without_llm.json",
                                        mime="application/json"
                                    )
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            st.error(f"Error reading or applying metadata file {metadata_file_path}: {e}")

    elif selected_scenario == "Benchmark_Constituent_with_relation":
        # Load necessary CSV files for multi-table processing
        benchmark_constituent_path = local_paths["Benchmark_Constituent_with_relation"][0]  # Benchmark_Constituent.csv
        benchmarks_csv_path = local_paths["Benchmark_Constituent_with_relation"][1]  # Benchmarks.csv
        securities_path = local_paths["Benchmark_Constituent_with_relation"][2]  # Securities.csv
        # Load DataFrames
        benchmark_constituent_df = pd.read_csv(benchmark_constituent_path)
        benchmarks_df = pd.read_csv(benchmarks_csv_path)
        securities_df = pd.read_csv(securities_path)
        # Add scale slider
        scale = st.sidebar.slider("Select Scale", min_value=1, max_value=100, value=5, step=5)
        metadata_file_path = local_rules["Benchmark_Constituent_with_relation"]
        try:
            with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                metadata_json = metadata_file.read()
                # Auto-detect metadata for multi-table setup
                metadata = detect_multi_table_metadata({
                    "Benchmark_Constituent": benchmark_constituent_df,
                    "Securities": securities_df,
                    "Benchmarks": benchmarks_df
                })
                # Apply metadata from JSON (update tables)
                apply_metadata_from_json(metadata, json.loads(metadata_json))
                if st.button("Generate Data"):
                    with st.spinner('Processing...'):
                        # Generate synthetic data for multi-table setup with scaling
                        metadata, synthetic_data = process_multi_table_Scenario({
                            "Benchmark_Constituent": benchmark_constituent_df,
                            "Securities": securities_df,
                            "Benchmarks": benchmarks_df
                        }, scale, metadata_json)
                        if synthetic_data is not None:
                            benchmark_constituent_synthetic = synthetic_data["Benchmark_Constituent"]
                            securities_synthetic = synthetic_data["Securities"]
                            benchmarks_synthetic = synthetic_data["Benchmarks"]
                            # Normalize the WEIGHT(%) column in the synthetic Benchmark Constituent data
                            benchmark_constituent_synthetic = normalize_weights(
                                benchmark_constituent_synthetic, weight_column='WEIGHT(%)'
                            )
                            # Display and download Benchmark Constituent synthetic data
                            st.write("Synthetic Benchmark Constituent Data:")
                            st.dataframe(benchmark_constituent_synthetic, hide_index=True)
                            constituent_csv = convert_df(benchmark_constituent_synthetic)
                            st.download_button(
                                label="Download Benchmark Constituent Synthetic Data CSV",
                                data=constituent_csv,
                                file_name="benchmark_constituent_synthetic_data.csv",
                                mime="text/csv"
                            )
                            # Display and download Securities synthetic data
                            st.write("Synthetic Securities Data:")
                            st.dataframe(securities_synthetic, hide_index=True)
                            securities_csv = convert_df(securities_synthetic)
                            st.download_button(
                                label="Download Securities Synthetic Data CSV",
                                data=securities_csv,
                                file_name="securities_synthetic_data.csv",
                                mime="text/csv"
                            )
                            # Display and download Benchmarks synthetic data
                            st.write("Synthetic Benchmarks Data:")
                            st.dataframe(benchmarks_synthetic, hide_index=True)
                            benchmarks_csv = convert_df(benchmarks_synthetic)
                            st.download_button(
                                label="Download Benchmarks Synthetic Data CSV",
                                data=benchmarks_csv,
                                file_name="benchmarks_synthetic_data.csv",
                                mime="text/csv"
                            )
                            # Download Metadata JSON
                            if metadata:
                                metadata_json = metadata_to_json(metadata)
                                st.sidebar.download_button(
                                    label="Download Metadata JSON",
                                    data=metadata_json,
                                    file_name="metadata.json",
                                    mime="application/json"
                                )
        except Exception as e:
            st.error(f"Error reading or applying metadata file {metadata_file_path}: {e}")
    else:
        if selected_scenario:
            scale = st.sidebar.slider("Select Scale", min_value=1, max_value=100, value=5, step=5)
        try:
            df = pd.read_csv(local_paths[selected_scenario])
            metadata_file_path = local_rules[selected_scenario]
            try:
                with open(metadata_file_path, "r", encoding='utf-8') as metadata_file:
                    metadata_json = metadata_file.read()
                    if st.button("Generate Data"):
                        with st.spinner('Processing...'):
                            metadata = detect_single_table_metadata(df)
                            apply_metadata_from_json(metadata, json.loads(metadata_json))
                            metadata, synthetic_data = process_single_table_Scenario(df, scale, metadata_json)
                            if synthetic_data is not None:
                                if selected_scenario == "Counter_Parties":
                                    df[['first_name', 'last_name']] = synthetic_data['CONTACT_PERSON'].str.split(expand=True)
                                    synthetic_data["EMAIL"] = (df['first_name'].str.lower() + '.' + df['last_name'].str.lower() + '@gmail.com')
                            
                                st.dataframe(synthetic_data, hide_index=True)
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