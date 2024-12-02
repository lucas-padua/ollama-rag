from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from . import chroma_database


class RAGPipeline:
    def __init__(self, documents_path):
        Settings.llm = Ollama(model="medllama2")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        documents = SimpleDirectoryReader(documents_path).load_data()
        self.index = chroma_database.build_index(documents)
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=6,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            node_postprocessor=[postproc],
        )

    def query(self, user_query):
        response = self.query_engine.query(user_query)
        return response.response
