import fire

class RAGCLI:
    """RAG System Command Line Interface"""

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.
        """
        print("[CLI TEST] index コマンドが呼ばれました。")
        print(f"[CLI TEST] max_chunk_size = {max_chunk_size} (型: {type(max_chunk_size).__name__})")

    def serach(self, query: str, k: int = 5) -> None:
        """
        Return the top-k sources for a single query.
        """
        print("[CLI TEST] index コマンドが呼ばれました。")
        print(f"[CLI TEST] query = {query} k = {k}")

def main() -> None:
    fire.Fire(RAGCLI)

if __name__ == "__main__":
    main()
