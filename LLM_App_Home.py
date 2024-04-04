# from dotenv import load_dotenv
# CHROMA_DB_PATH = "./chroma_vector_database"
# load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')