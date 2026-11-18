import bm25s
import json

# BM25クラスが持つメソッド・属性の一覧を表示（_で始まる内部用を除く）
# print([m for m in dir(bm25s.BM25) if not m.startswith("_")])
#
# # 特定のメソッドの引数と説明（docstring）を表示
# help(bm25s.BM25.retrieve)
# help(bm25s.tokenize)

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
        print(f"file_path: {corpus_data[cid]['file_path']}")
        print(f"first_character_index: {corpus_data[cid]['first_character_index']}")
        print(f"last_character_index: {corpus_data[cid]['last_character_index']}")
        # print(f"TEXT: {corpus_data[cid]['text']}")
        print()


