from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.llms.ai21 import AI21

from data_processing.vector_database import load_vector_database
from llama_custom.retriever import VectorDBRetriever
from llama_custom.query_engine import RAGStringQueryEngine, RAGQueryEngine
from llama_custom.response_synthesizer import RAGCompactAndRefine

import os
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = ('/').join(ROOT_DIR[:-1])

parser = argparse.ArgumentParser()
parser.add_argument('--db', default=f'{ROOT_DIR}/evaluation/truthful_db', type=str)
parser.add_argument('--query', default='Which country drinks the most coffee?', type=str)
parser.add_argument('--engine', default='rag_citations', type=str)
parser.add_argument('--local', default=False, type=bool)

# AI21 api key
api_key = "XcXwkV8LXRtuoqZdjFWIwS3fdGx1fS83"


def get_embed_model():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    return embed_model


def load_llm_model(model_path=None):
    model_url = None
    if model_path is None:
        model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # set the path to a pre-downloaded model instead of model_url
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=128,     #256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=2500,    #3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        verbose=True,
    )

    return llm


def load_llm_model_cloud():
    llm = AI21(api_key=api_key)
    return llm


# synthesize a response using llm and retriever
def get_rag_engine(retriever, llm, response_synthesizer=None):
    query_engine = RAGQueryEngine.from_args(retriever, llm=llm, response_synthesizer=response_synthesizer)
    return query_engine


def get_rag_string_engine(retriever, llm):
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    query_engine = RAGStringQueryEngine(retriever=retriever, llm=llm, qa_prompt=qa_prompt)
    return query_engine


def get_retriever_engine(retriever, llm):
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    return query_engine


def get_query_engine(engine, retriever, llm):
    if engine == "rag_citations":
        # This engine incorporates context info in response
        query_engine = get_rag_string_engine(retriever, llm)

    elif engine == "simple_rag":
        response_synthesizer = RAGCompactAndRefine(llm=llm)
        query_engine = get_rag_engine(retriever, llm, response_synthesizer)

    else:
        # use retriever engine
        query_engine = get_retriever_engine(retriever, llm)
    return query_engine


def get_query_response(query_str, query_engine):
    # Takes about 30min for locally-run llms even on increased swap!
    # query: retrieves, then response synthesis
    response = query_engine.query(query_str)
    return response


def main():
    args = parser.parse_args()
    engine = args.engine

    # Example Query for txtai_database_vector
    # query_str = "What are the key concepts of txtai?"
    # Example queries for Llama_db
    # query_str = "Can you tell me about the key concepts for safety finetuning"
    # query_str = "How does Llama 2 perform compared to other open-source models?"
    # Example query for truthful dataset:
    #query_str = "Americans drink more coffee than people in which other countries?"

    embed_model = get_embed_model()

    if args.local:
        #model_path = "../llama_index/models/llama-2-13b-chat.Q4_0.gguf"
        llm = load_llm_model()
    else:
        llm = load_llm_model_cloud()

    vector_store = load_vector_database(args.db)

    # use a Retriever to retrieve sth from results
    # by default llama-index uses cosine similarity
    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=2
    )

    query_engine = get_query_engine(engine, retriever, llm)

    # run a query
    print('Q: ', args.query)
    response = get_query_response(args.query, query_engine)
    print('A: ', str(response))


if __name__ == "__main__":
    main()

