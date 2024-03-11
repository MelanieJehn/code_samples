import os
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
import chromadb
import argparse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_custom.retriever import VectorDBRetriever

from llama_custom.rag_application import get_embed_model, get_query_engine, load_llm_model_cloud

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = ('/').join(ROOT_DIR[:-1])

index = None
query_engine = None
lock = Lock()

parser = argparse.ArgumentParser()
parser.add_argument('--db', default=f'{ROOT_DIR}/evaluation/truthful_db', type=str)
parser.add_argument('--engine', default='rag_citations', type=str)


def initialize_index(db_name, collection=None):
    global index

    with lock:
        chroma_db = chromadb.PersistentClient(path=db_name)
        if collection is None:
            collection = db_name.split('/')[-1]
        chroma_collection = chroma_db.get_or_create_collection(collection)

        # set up ChromaVectorStore and load in data
        index = ChromaVectorStore(chroma_collection=chroma_collection)


def query_index(query_text):
    global index
    response = query_engine.query(query_text)
    return str(response)


def main():
    args = parser.parse_args()
    engine = args.engine
    print("initializing index...")
    embed_model = get_embed_model()
    # easy working example: db_name = "llama_db" and collection = "llama_nodes"
    #initialize_index(f"{ROOT_DIR}/data_processing/llama_db", collection="llama_nodes")
    initialize_index(args.db)
    retriever = VectorDBRetriever(
        index, embed_model, query_mode="default", similarity_top_k=2
    )
    llm = load_llm_model_cloud()
    global query_engine
    query_engine = get_query_engine(engine, retriever, llm)
    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(("", 5602), b"password")
    manager.register("query_index", query_index)
    server = manager.get_server()

    print("starting server...")
    server.serve_forever()


if __name__ == "__main__":
    # init the global index
    main()
