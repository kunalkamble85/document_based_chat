import streamlit as st
from utils.oci_utils import *
from PIL import Image
from utils.langchain_utils import get_data_from_chat_db, display_sidebar
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

st.set_page_config(page_title="Talk to Database", page_icon=":book:", layout="wide")
display_sidebar()
st.title("🤖 Talk to Database")


def handle_conversation():
    conversation = st.session_state["chat_db_history"]
    response, df = get_data_from_chat_db(conversation)
    return response, df

if "chat_db_messages" not in st.session_state:
    st.session_state["chat_db_messages"] = [{"role":"assistant" , "content":("Ask your question to fetch data from customer, accounts and monthly_records tables", None)}]
if "chat_db_history" not in st.session_state:
    st.session_state.chat_db_history = []

with st.expander("Show table description"):
    st.info("""customers (Customer_ID VARCHAR, Name VARCHAR, Age INTEGER, SSN VARCHAR, Occupation VARCHAR, Annual_Income FLOAT)""")
    st.info("accounts (Customer_ID VARCHAR, Num_Bank_Accounts INTEGER, Num_Credit_Card INTEGER, Num_of_Loan INTEGER, Type_of_Loan VARCHAR)")
    st.info("monthly_records (ID VARCHAR, Customer_ID VARCHAR, Month VARCHAR, Monthly_Inhand_Salary FLOAT, Delay_from_due_date INTEGER, Num_of_Delayed_Payment INTEGER, Changed_Credit_Limit FLOAT, Num_Credit_Inquiries INTEGER, Credit_Mix VARCHAR, Outstanding_Debt FLOAT, Credit_Utilization_Ratio FLOAT, Credit_History_Age VARCHAR, Payment_of_Min_Amount VARCHAR, Total_EMI_per_month FLOAT, Amount_invested_monthly FLOAT, Payment_Behaviour VARCHAR, Monthly_Balance FLOAT)")

sample_selected = False
sample_query = ""
with st.expander("Sample Questions"):    
    sample1 = st.button("What is the average age of the customers?")
    if sample1:
        sample_query = "What is the average age of the customers?"
        sample_selected = True
    sample2 = st.button("Who are the top 5 oldest customers?")
    if sample2:
        sample_query = "Who are the top 5 oldest customers?"
        sample_selected = True
    sample3 = st.button("Which customer makes the largest average monthly payment?")
    if sample3:
        sample_query = "Which customer makes the largest average monthly payment?"
        sample_selected = True
    sample4 = st.button("Create a binning of annual salary and analyze that against the average number of credit cards?")
    if sample4:
        sample_query = "Create a binning of annual salary and analyze that against the average number of credit cards?"
        sample_selected = True

clear_history = st.button("Clear conversation history")
if clear_history:
    st.session_state["chat_db_messages"] = [{"role":"assistant" , "content":("Ask your question to fetch data from customer, accounts and monthly_records tables", None)}]
    st.session_state["chat_db_history"] = []

for message in st.session_state["chat_db_messages"]:
    # with st.chat_message(message["role"]):
    if message["role"] == "user":
        st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(message["content"])
    else:
        content = message["content"][0]
        if content == "Ask your question to fetch data from customer, accounts and monthly_records tables":
            st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(content)
        else:
            with st.expander("SQL query", expanded=True):       
                st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(content)
        df = message["content"][1]
        if df is not None:
            st.dataframe(data = df, hide_index=True)
        # st.write(message["content"])

def show_data(df):
    if df is not None:
        columns = df.columns
        df = df[:100]
        width = 300 
        height = 300

        x_axis = columns[0]
        numeric_columns = [col for col in columns if is_numeric_dtype(df[col]) and x_axis!=col]

        df_grouped = df.groupby(x_axis).sum()

        if len(numeric_columns) > 0:
            tab0, tab1, tab2, tab3, tab4= st.tabs(["Data", "Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart"])
            with tab0:
                st.title("Data")
                st.dataframe(data = df, hide_index=True)
            with tab1:
                st.title("Line Chart")
                st.line_chart(df_grouped[numeric_columns], width=width, height=height, use_container_width=True)
            with tab2:
                st.title("Bar Chart")
                st.bar_chart(df_grouped[numeric_columns], stack=False)
            with tab3:
                st.title("Pie Chart")            
                fig, ax = plt.subplots()             
                df_grouped[numeric_columns[0]].plot.pie(autopct='%1.1f%%', ax=ax)
                st.pyplot(fig, use_container_width=False)
            with tab4:
                st.title("Scatter Chart")
                st.scatter_chart(df_grouped[numeric_columns])            
        else:
            st.dataframe(data = df, hide_index=True)


if user_query := st.chat_input("Enter your question here") or sample_selected:
    if sample_selected: user_query = sample_query
    st.session_state["chat_db_messages"].append({"role":"user","content":user_query})
    st.chat_message("user", avatar=Image.open('./images/chat_user.jpeg')).write(user_query)
    with st.spinner("Thinking.."):
        st.session_state["chat_db_history"].append({"role":"user","content":user_query})
        response, df = handle_conversation()
        st.session_state["chat_db_history"].append({"role":"assistant","content":response})
        with st.expander("SQL query", expanded=False):            
            st.chat_message("assistant", avatar=Image.open('./images/ai_robot.jpg')).write(response)
        if df is not None:
            show_data(df)
        else:
            st.info("No Data found.")
        st.session_state["chat_db_messages"].append({"role":"assistant","content":(response, df)})