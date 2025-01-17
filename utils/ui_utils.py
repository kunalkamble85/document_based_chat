import streamlit as st
import base64
from pathlib import Path

def display_sidebar():
    custom_css = """
    <style>
        [data-testid=stSidebar] {
            background-color: #DEDDDD !important;
        }
    </style>
    """
    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    # Streamlit app content, including the sidebar
    with st.sidebar:
        pass
    with st.sidebar:
        logo = f"url(data:image/png;base64,{base64.b64encode(Path('./images/oracle_logo.jpg').read_bytes()).decode()})"
        st.markdown(
            f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: {logo};
                    background-repeat: no-repeat;
                    padding-top: 180px;
                    background-size: 290px 120px;
                    background-position: 20px 20px;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.write("""<div style="width:100%;text-align:left">
                <br style="font-size: 1em;"><b>Powered by</b>
                <br style="font-size: 3em; font-weight: bold;"><b><u>OCI Generative AI</u></b>    
                </div>      
                """, unsafe_allow_html=True)
        
        st.write("""<div style="width:100%;text-align:left">
                <br style="font-size: 1em;"><b>Built by</b>
                <br style="font-size: 3em; font-weight: bold;"><b><u>Finergy AI Team</u></b>    
                </div>      
                """, unsafe_allow_html=True)