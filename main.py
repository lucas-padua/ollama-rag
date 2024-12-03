from src.rag_pipeline import RAGPipeline
import sys
import time


def main():
    rag = RAGPipeline("Documents")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        rag.query(query)


if __name__ == "__main__":
    main()
