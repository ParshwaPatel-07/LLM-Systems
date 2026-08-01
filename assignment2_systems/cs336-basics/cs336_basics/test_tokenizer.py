import time
from tokenizer import BPETokenizer

tokenizer = BPETokenizer.from_files(
    vocab_path="vocab.pkl",
    merge_path="merges.pkl",
    special_tokens=["<|endoftext|>"],
)

with open("TinyStories-250M.txt", "r", encoding="utf-8") as f:
    text = f.read(1_000_000)   # 1 MB

start = time.perf_counter()
ids = tokenizer.encode(text)
end = time.perf_counter()

print(f"Time: {end-start:.2f} seconds")
print(f"Tokens: {len(ids)}")