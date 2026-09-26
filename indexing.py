from pathlib import Path
from rank_bm25 import BM25Okapi
# First import the chunker you want from Chonkie
from chonkie import RecursiveChunker, CodeChunker
import re
from loading import get_file_path

# # Initialize the chunker
# chunker = RecursiveChunker(
#     tokenizer = "character",
#     chunk_size = 200
# )
#
# code_chunker = CodeChunker(
#     language="python",      # Specify the programming language
#     tokenizer="character",  # Default tokenizer (or use "gpt2", etc.)
#     chunk_size=300,        # Maximum tokens per chunk
#     include_nodes=False     # Optionally include AST nodes in output
# )
# # Chunk some text
# with open("vllm-0.10.1/README.md", encoding="utf-8") as f:
#     text = f.read()
# chunks = chunker(text)
#
# with open("vllm-0.10.1/vllm/engine/protocol.py", encoding="utf-8") as f:
#     text = f.read()
# chunks_code = code_chunker(text)
#
# # Access chunks
# for chunk in chunks:
#     print(f"Chunk: {chunk.text}")
#     print(f"Tokens: {chunk.token_count}")
#     print(f"Start Index: {chunk.start_index}")
#     print(f"End Index: {chunk.end_index}")
#     print("================================================================")


# for chunk in chunks_code:
#     print(f"Chunk: {chunk.text}")
#     print(f"Tokens: {chunk.token_count}")
#     print("================================================================")
#
dir_path = Path("vllm-0.10.1")
extension = {".py", ".md"}

for file_path in get_file_path(dir_path, extension):
    print(file_path)


# bm25の動作テストであり本番実装ではない
# 事前に単語分割(トークン化)したリストのリストを渡す
chunk_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "FOX! FOX! FOX! The Fox is very fast.",
        "Artificial Intelligence and Machine Learning! are changing the world.",
        "Hello world"
]
# chunk_split = [text.lower().split() for text in chunk_texts]  # 英語なら空白区切りでOK

tokenizer_pattern = re.compile(r"[a-z0-9_]+")
def tokenize(text: str) -> list[str]:
    return tokenizer_pattern.findall(text.lower())
tokenized_corpus = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

query = "machine learning"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)  # 各チャンクとのスコア(numpy配列)

top_k_indices = scores.argsort()[::-1][:10]  # 上位10件のインデックス
print(top_k_indices)
