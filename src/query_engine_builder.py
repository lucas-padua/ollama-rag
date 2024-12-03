from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


def build_query_engine(index):
    retriever = VectorIndexRetriever(index, similarity_top_k=3)

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        response_mode="tree_summarize",
    )
    return query_engine
