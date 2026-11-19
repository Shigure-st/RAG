import bm25s
import json


def run_retriever(query: str, k: int) -> None:
    retriever = bm25s.BM25.load("data/processed/bm25_index")
    query_tokens = bm25s.tokenize([query])

    results, scores = retriever.retrieve(query_tokens, k=k)
    chunk_ids = results[0]  # 1クエリ分の結果
    print(results)
    print(scores)

    with open("data/processed/chunk_index", "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    # print(chunk_ids)
    for cid in chunk_ids:
        print(
            f"file_path: {corpus_data[cid]['file_path']}"
            f"[{corpus_data[cid]['first_character_index']}"
            f":{corpus_data[cid]['last_character_index']}]"
        )
        # print(f"TEXT: {corpus_data[cid]['text']}")
        print()


