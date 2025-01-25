from llama_index.core import (
    SimpleDirectoryReader,
    PromptTemplate,
)

from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.evaluation import FaithfulnessEvaluator
from . import chroma_database
import sys
import time


class RAGPipeline:
    def __init__(self, documents_path):
        Settings.llm = Ollama(model="medllama2", request_timeout=600)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        documents = SimpleDirectoryReader(documents_path).load_data()
        self.index = chroma_database.build_index(documents)
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

        qa_prompt_template = """
        You are a Gynecologist, and your expertise is in endometriosis. \n
        Respond to the query with the provided documents. \n
        Query: {query_str}. \n
        Be concise and provide reference if possible.
        """
        qa_prompt = PromptTemplate(qa_prompt_template)

        response_synthesizer = TreeSummarize(summary_template=qa_prompt)
        self.evaluator = FaithfulnessEvaluator(llm=Settings.llm)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=6,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            node_postprocessor=[postproc, rerank],
            response_synthetizer=response_synthesizer,
        )

    def query(self, user_query):
        return self.query_engine.query(user_query)
        # evaluation = self.evaluator.evaluate_response(response=response)
        # for token in str(response).split(" "):
        #     print(token, end=" ", flush=True)
        #     sys.stdout.flush()
        #     time.sleep(0.4)
        # print()
        # print("\n", str(evaluation.passing))
        # print("\n", str(evaluation.feedback))
