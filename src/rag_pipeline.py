from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# from llama_index.core.postprocessor import (
#     MetadataReplacementPostProcessor,
#     SentenceTransformerRerank,
# )
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.query_pipeline import QueryPipeline
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

        prompt_string = """You are a doctor, a specialist in endometriosis, that gives direct and concise answers,
        never gives medical advice and only answers something when the response is found in the context.
        Given the context and not prior knowledge, answer the query: {query_str}"""
        prompt_template = PromptTemplate(prompt_string)

        self.pipeline = QueryPipeline(chain=[prompt_template, Settings.llm])
        self.evaluator = FaithfulnessEvaluator(llm=Settings.llm)

    def query(self, user_query):
        response = self.pipeline.run(user_query)
        print(response)
        # evaluation = self.evaluator.evaluate_response(
        #     response=response, query=user_query, contexts=response.source_nodes
        # )
        # for token in str(response).split(" "):
        #     print(token, end=" ", flush=True)
        #     sys.stdout.flush()
        #     time.sleep(0.4)
        # print()
        # print("\n", str(evaluation.passing))
        # print("\n", str(evaluation.feedback))
