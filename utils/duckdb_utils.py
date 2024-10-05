import duckdb

def execute_query(query):
    conn = duckdb.connect('./chat_db/credit_scoring.duckdb')
    query_result = conn.execute(query).df()
    conn.close()
    return query_result

# df = execute_query("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
# print(df.head())
# print(df.columns)