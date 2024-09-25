import os
import sys
import hashlib
import base64
import numpy as np
from datetime import datetime
import docx2txt
from typing import List
from openai import AzureOpenAI
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.config import cfg

DOCUMENTS_BASE_PATH = ''
VECTODB_BASE_PATH = ''

client = AzureOpenAI(
    api_key=cfg.AZURE_OPENAI_API_KEY,
    api_version=cfg.API_VERSION,
    azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT
)

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return list(x / norm)
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return list(np.where(norm == 0, x, x / norm))


def get_embedding(text: str, engine: str = "text-embedding-3-small") -> List:
    text = text.strip().replace("\n", " ")
    response = client.embeddings.create(input=[text],
                                        model=engine,
                                        dimensions=cfg.DIMENSION,
                                        )
    return normalize_l2(response.data[0].embedding)
    

def create_unique_id(string: str) -> str:
    """
        Unique ID creation per news for database primary key.
    """
    sha256 = hashlib.sha256()
    sha256.update(string.encode('utf-8'))
    hash_digest = sha256.digest()
    encoded = base64.b64encode(hash_digest, altchars=b'-_').decode('utf-8')
    alphanumeric = ''.join(char for char in encoded if char.isalnum()).upper()
    return alphanumeric[:10]

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def split_text_to_docs_list(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    docs_list = text_splitter.create_documents([text])
    return docs_list


def create_vector_db_and_collection(collection_name="Library_1"):
    client = chromadb.PersistentClient(path=f"{VECTODB_BASE_PATH}/vectordb")
    embed_func = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=cfg.AZURE_OPENAI_API_KEY,
        api_base=cfg.AZURE_OPENAI_ENDPOINT,
        api_version=cfg.API_VERSION,
        api_type='azure',
    )
    collection = client.get_or_create_collection(
        name=f"{collection_name}",
        embedding_function=embed_func
    )
    return collection


def load_docx_files_into_db(collection=None, collection_name="Library_1"):
    data_dir = f"{DOCUMENTS_BASE_PATH}/{collection_name}"
    
    for i, file_name in enumerate(os.listdir(data_dir)):
        if not file_name.endswith('.docx'):
            continue

        filepath = os.path.join(data_dir, file_name)
        text = extract_text_from_docx(filepath)
        text_splits = split_text_to_docs_list(text)
        text_splits = [text_split.page_content for text_split in text_splits]

        _ids = [f"{collection_name}/{file_name}-split_{i}" for i in range(len(text_splits))]
        _ids = [create_unique_id(id) for id in _ids]
        
        embedding_list = [get_embedding(text) for text in text_splits]
        metadata = [{
            'collection': collection_name,
            'filepath': f"./{collection_name}/{file_name}",
            'split_id': k+1,
            'timestamp': datetime.today().strftime('%Y-%m-%d'),
        } for k in range(len(text_splits))]

        print(f"file={file_name}, splits={len(text_splits)}")
        collection.upsert(ids=_ids,
                          embeddings=embedding_list,
                          metadatas=metadata,
                          documents=text_splits
                          )
    print(f"{i+1} files are indexed and added to the collection {collection_name}!")


# create vectordb and add collection
if __name__ == '__main__':
      collection = create_vector_db_and_collection(collection_name="Library_1")
      load_docx_files_into_db(collection, collection_name="Library_1")
  
      _collection = create_vector_db_and_collection(collection_name="Library_2")
      load_docx_files_into_db(_collection, collection_name="Library_2")
