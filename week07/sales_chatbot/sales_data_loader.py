import yaml
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class DataConf:
    def __init__(self, vector_store_dir: str, data_file: str):
        self.vector_store_dir = vector_store_dir
        self.data_file = data_file

def load_data(configs):
    text_splitter = CharacterTextSplitter(        
        separator = r'\n\n',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    for c in configs:
        with open(c.data_file, 'r') as f:
            real_estate_sales = f.read()
            docs = text_splitter.create_documents([real_estate_sales])
            print(f"save {len(docs)} docs to {c.vector_store_dir}")
            db = FAISS.from_documents(docs, OpenAIEmbeddings())
            db.save_local(c.vector_store_dir)

def load_data_configs(path="conf.yaml"):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        result = []
        for c in data['bots']:
            vector_store_dir = c['vector_store_dir']
            data_file = c['data_file'] if 'data_file' in c else f"{c['vector_store_dir']}_data.txt"
            result.append(DataConf(vector_store_dir, data_file))            
        return result

if __name__ == "__main__":
    configs = load_data_configs()
    load_data(configs)
