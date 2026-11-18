import fire
from src.indexing import run_indexing
from src.retriever import run_retriever


class RAGCLI:
    """RAG System Command Line Interface"""

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.
        """
        run_indexing(max_chunk_size=int(max_chunk_size))

    def search(self, query: str, k: int = 5) -> None:
        """
        Return the top-k sources for a single query.
        """
        print("[CLI TEST] index コマンドが呼ばれました。")
        print(f"[CLI TEST] query = {query} k = {k}")
        run_retriever(query, k=int(k))


def main() -> None:
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    main()
