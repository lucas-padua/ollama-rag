from src.rag_pipeline import RAGPipeline


def main():
    rag = RAGPipeline("Documents")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        response = rag.query(query)
        print("Assistant:", response)


if __name__ == "__main__":
    main()
