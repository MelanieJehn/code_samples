from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llama_dataset import (
    LabelledRagDataset,
    download_llama_dataset
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

import chromadb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--db', default='./truthful_512_db', type=str)
parser.add_argument('--dataset_name', default='MiniTruthfulQADataset', type=str)
parser.add_argument('--docs', default='"./data/source_files"', type=str)


def create_vector_database(embed_model, documents, db_name, coll_name=None):
    """
    Creates a Chroma vector database with name "db_name" and adds embeddings of "documents"
    :param embed_model: Embedding model for text
    :param db_name: name of the database to be created
    :param coll_name: name of the database collection
    :return: vector database object (ChromaVectorStore)
    """
    # Here we decide the size of text chunks and overlap between chunks!
    # chunk_size originally=1024 for truthful_db and all others
    text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in query responses
    # doc_idxs has #chunks in doc entries per doc
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    # construct text nodes from text chunks
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    # embed each node keeping the metadata
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    # create client and a new collection
    chroma_db = chromadb.PersistentClient(path=f"./{db_name}")
    if coll_name is None:
        coll_name = db_name
    chroma_collection = chroma_db.create_collection(coll_name)

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_store.add(nodes)
    #storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    return vector_store


def load_vector_database(db_path='./txtai_database', collection=None):
    """
    Load a Chroma vector database from disk
    :param db_name: the name of the database to be loaded
    :param collection: the name of the corresponding collection
    :return: a vector database object (ChromaVectorStore)
    """
    # Created datasets: llama_db with coll llama_nodes
    #                   txtai_database_vector with coll txtai_testdata
    chroma_db = chromadb.PersistentClient(path=db_path)
    if collection is None:
        collection = db_path.split('/')[-1]
    chroma_collection = chroma_db.get_or_create_collection(collection)
    print(f'Getting collection: {collection} from path {db_path}')

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    #index = VectorStoreIndex.from_vector_store(
     #   vector_store,
     #   embed_model=embed_model,
    #)

    return vector_store


def generate_dataset_from_docs(llm, documents):
    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        llm=llm,
        num_questions_per_chunk=10,  # set the number of questions per nodes
    )

    rag_dataset = dataset_generator.generate_dataset_from_nodes()
    return rag_dataset


def get_llama_rag_dataset(dataset_name="PaulGrahamEssayDataset"):
    rag_dataset, documents = download_llama_dataset(
        dataset_name, "./data"
    )
    return rag_dataset, documents


def create_db_and_rag_dataset(embed_model, dataset_name, db):
    rag_dataset, docs = get_llama_rag_dataset(dataset_name)
    vector_store = create_vector_database(embed_model, docs, db)
    return vector_store


def main():
    args = parser.parse_args()
    # Create a new database from documents on disk or download documents
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    if args.docs:
        # we already have docs available -> only create vector databaase
        docs = SimpleDirectoryReader(args.docs).load_data()
        vector_store = create_vector_database(embed_model, docs, args.db)
    else:
        vector_store = create_db_and_rag_dataset(embed_model, args.dataset_name, args.db)


if __name__ == "__main__":
    main()
