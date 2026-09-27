from pathlib import Path
from rank_bm25 import BM25Okapi
import re
from loading import get_file_path
from chonkie import RecursiveChunker, CodeChunker
import json
import bm25s


text_chunker = RecursiveChunker(
    tokenizer = "character",
    chunk_size = 1000
)

code_chunker = CodeChunker(
    language="python",      # Specify the programming language
    tokenizer="character",  # Default tokenizer (or use "gpt2", etc.)
    chunk_size=1000,        # Maximum tokens per chunk
    include_nodes=False     # Optionally include AST nodes in output
)

dir_path = Path("vllm-0.10.1")
extension = {".py", ".md"}
chunk_id = 0
all_chunks = []

for file_path in get_file_path(dir_path, extension):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    if file_path.suffix == ".py":
        chunks = code_chunker(text)
    else:
        chunks = text_chunker(text)
    for chunk in chunks:
        # print("================================================================")
        # print(f"Chunk: {chunk.text}")
        # print(f"Tokens: {chunk.token_count}")
        # print(f"Start Index: {chunk.start_index}")
        # print(f"End Index: {chunk.end_index}")
        # print("================================================================")
        chunk_data = {
            "chunk_id": chunk_id,
            "file_path": str(file_path),
            "first_character_index": chunk.start_index,
            "last_character_index": chunk.end_index,
            "text": chunk.text
        }
        all_chunks.append(chunk_data)
        chunk_id += 1

with open("data/processed/chunk_index", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)


with open("data/processed/chunk_index", "r", encoding="utf-8") as f:
    corpus_data = json.load(f)
corpus_texts = [data["text"] for data in corpus_data]
corpus_tokens = bm25s.tokenize(corpus_texts)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)
retriever.save("data/processed/bm25_index")

# with open("data/processed/chunks.json", encoding="utf-8") as f:
#     chunks = json.load(f)  # chunk_id == リストのインデックス、という前提

query_tokens = bm25s.tokenize("Hello World")
results, scores = retriever.retrieve(query_tokens, k=10)
chunk_ids = results[0]  # 1クエリ分の結果
print(results)
print(scores)
print(chunk_ids)
for cid in chunk_ids:
    print(f"file_path: {corpus_data[cid]['file_path']}")
    print(f"first_character_index: {corpus_data[cid]['first_character_index']}")
    print(f"last_character_index: {corpus_data[cid]['last_character_index']}")
    print(f"TEXT: {corpus_data[cid]['text']}")
    print()
    print()
    print()



# bm25の動作テストであり本番実装ではない
# 事前に単語分割(トークン化)したリストのリストを渡す
# chunk_texts = [
#         "The quick brown fox jumps over the lazy dog.",
#         "FOX! FOX! FOX! The Fox is very fast.",
#         "Artificial Intelligence and Machine Learning! are changing the world.",
#         "Hello world"
# ]
# # chunk_split = [text.lower().split() for text in chunk_texts]  # 英語なら空白区切りでOK
#
# tokenizer_pattern = re.compile(r"[a-z0-9_]+")
# def tokenize(text: str) -> list[str]:
#     return tokenizer_pattern.findall(text.lower())
# tokenized_corpus = [tokenize(text) for text in chunk_texts]
# bm25 = BM25Okapi(tokenized_corpus)
#
# query = "machine learning"
# tokenized_query = query.lower().split()
# scores = bm25.get_scores(tokenized_query)  # 各チャンクとのスコア(numpy配列)
#
# top_k_indices = scores.argsort()[::-1][:10]  # 上位10件のインデックス
# print(top_k_indices)
