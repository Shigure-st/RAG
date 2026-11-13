from pathlib import Path
from rank_bm25 import BM25Okapi
import re
from src.loading import get_file_path
from chonkie import RecursiveChunker, CodeChunker
import json
import bm25s
from typing import List
import numpy as np
from fastembed import TextEmbedding
import sys
import os
from tqdm import tqdm



def run_indexing(max_chunk_size: int = 2000):
    text_chunker = RecursiveChunker(
        tokenizer = "character",
        chunk_size = max_chunk_size
    )

    code_chunker = CodeChunker(
        language="python",      # Specify the programming language
        tokenizer="character",  # Default tokenizer (or use "gpt2", etc.)
        chunk_size=max_chunk_size,        # Maximum tokens per chunk
        include_nodes=False     # Optionally include AST nodes in output
    )

    embedding_model = TextEmbedding(
        model_name = "minishlab/potion-base-8M",
        threads=6
    )

    dir_path = Path("vllm-0.10.1")
    extension = {".py", ".md"}
    chunk_id = 0
    all_chunk_text = []
    all_chunks = []

    for file_path in tqdm(get_file_path(dir_path, extension), desc="chunk分割中"):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        if file_path.suffix == ".py":
            chunks = code_chunker(text)
        else:
            chunks = text_chunker(text)
        current_start = None
        current_end = None
        for chunk in chunks:
            if current_start is None:
                current_start = chunk.start_index
                current_end = chunk.end_index
                continue

            potential_size = chunk.end_index - current_start
            if potential_size <= max_chunk_size / 2:
                current_end = chunk.end_index
            else:
                if current_end - current_start > max_chunk_size:
                    current_end = current_start + max_chunk_size
                merged_text = text[current_start:current_end]
                chunk_data = {
                    "chunk_id": chunk_id,
                    "file_path": str(file_path),
                    "first_character_index": current_start,
                    "last_character_index": current_end,
                    "text": merged_text
                }
                all_chunk_text.append(merged_text)
                all_chunks.append(chunk_data)
                chunk_id += 1

                current_start = chunk.start_index
                current_end = chunk.end_index

        if current_start is not None:
            if current_end - current_start > max_chunk_size:
                current_end = current_start + max_chunk_size
            merged_text = text[current_start:current_end]
            chunk_data = {
                "chunk_id": chunk_id,
                "file_path": str(file_path),
                "first_character_index": current_start,
                "last_character_index": current_end,
                "text": merged_text
            }
            all_chunk_text.append(merged_text)
            all_chunks.append(chunk_data)
            chunk_id += 1

    embeddings_generator = embedding_model.embed(all_chunk_text)
    vector_text = np.array(list(tqdm(
                    embeddings_generator,
                    total=chunk_id,
                    desc="ベクトル変換中")))
    np.save("data/processed/potion-8m_index.npy", vector_text)

    corpus_tokens = bm25s.tokenize(all_chunk_text)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save("data/processed/bm25_index")


    # with open("data/processed/chunk_index", "w", encoding="utf-8") as f:
    #     json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    #

# with open("data/processed/chunk_index", "r", encoding="utf-8") as f:
#     corpus_data = json.load(f)
# corpus_texts = [data["text"] for data in corpus_data]
# corpus_tokens = bm25s.tokenize(corpus_texts)
#
# retriever = bm25s.BM25()
# retriever.index(corpus_tokens)
# retriever.save("data/processed/bm25_index")

# with open("data/processed/chunks.json", encoding="utf-8") as f:
#     chunks = json.load(f)  # chunk_id == リストのインデックス、という前提
#
# query_tokens = bm25s.tokenize("Hello World")
# results, scores = retriever.retrieve(query_tokens, k=10)
# chunk_ids = results[0]  # 1クエリ分の結果
# print(results)
# print(scores)
# print(chunk_ids)
# for cid in chunk_ids:
#     print(f"file_path: {corpus_data[cid]['file_path']}")
#     print(f"first_character_index: {corpus_data[cid]['first_character_index']}")
#     print(f"last_character_index: {corpus_data[cid]['last_character_index']}")
#     print(f"TEXT: {corpus_data[cid]['text']}")
#     print()
#     print()
#     print()



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
