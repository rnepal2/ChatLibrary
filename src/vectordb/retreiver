import os, sys
import chromadb
from chromadb.utils import embedding_functions
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.config import cfg

class Conf(cfg):
    VECRODB_PATH = "/chromadb"
    

class VectorDBRetriever:
    '''Retriever class to fetch documents from collections for a query/question.'''
    def __init__(self, collection_list: list, n_results: int=5) -> None:
        self.collection_list = collection_list
        self.n_results = n_results

    def load_collection(self, collection_name: str):
        client = chromadb.PersistentClient(path=Conf.VECRODB_PATH)
        embed_func = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=Conf.AZURE_OPENAI_API_KEY,
            api_base=Conf.AZURE_OPENAI_ENDPOINT,
            api_version=Conf.API_VERSION,
            api_type=Conf.API_TYPE,
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_func
        )
        return collection

    def retrieve_documents(self, question: str):
        documents = []
        for collection in self.collection_list:
            coll = self.load_collection(collection_name=collection)
            if self.n_results > coll.count():
                self.n_results = coll.count()
            response = coll.query(query_texts=question,
                                        where=None,
                                        where_document=None,
                                        n_results=self.n_results,
                                   )
            documents.append(response)
        return documents
            

if __name__ == '__main__':
    query = 'What is ....?'
    retriever = VectorDBRetriever(collection_list=['Library_1', 'Library_2'], n_results=5)
    ans = retriever.retrieve_documents(query)

    for item in ans:
        print(item)
        print('\n='*50)
