from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def build_node_parser(documents: list) -> list:
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
        include_prev_next_rel=True,
    )
    return node_parser.get_nodes_from_documents(documents)


def build_chroma_vector_store():
    chroma_client = chromadb.PersistentClient(path="/var/chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("medical_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return StorageContext.from_defaults(vector_store=vector_store)


def build_index(documents: list):
    storage_context = build_chroma_vector_store()
    nodes = build_node_parser(documents)
    return VectorStoreIndex(nodes, storage_context=storage_context)
