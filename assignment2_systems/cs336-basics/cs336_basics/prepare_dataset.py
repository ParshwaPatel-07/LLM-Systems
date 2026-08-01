from pathlib import Path
import numpy as np
import time

from tokenizer import BPETokenizer

# Load your tokenizer
tokenizer = BPETokenizer.from_files(
    vocab_path="vocab.pkl",
    merge_path="merges.pkl",
    special_tokens=["<|endoftext|>"],
)

ROOT = Path(__file__).resolve().parent.parent.parent

datasets = [
    (ROOT / "data" / "TinyStoriesV2-GPT4-train.txt", "train.bin"),
    (ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt", "val.bin"),
]
start = time.perf_counter()
for input_file, output_file in datasets:

    print(f"Encoding {input_file}...")

    total_tokens = 0

    with open(output_file, "wb") as out:
        with open(input_file, "r", encoding="utf-8") as f:

            for i, line in enumerate(f):

                ids = np.array(
                    tokenizer.encode(line),
                    dtype=np.uint16,
                )

                ids.tofile(out)

                total_tokens += len(ids)

                if i % 10000 == 0:
                    print(f"{i:,} lines processed")

    print(f"Number of tokens: {total_tokens:,}")
    print(f"Saved {output_file}\n")
end = time.perf_counter()
print(f"Time: {end-start:.2f} seconds")
import numpy as np

train = np.memmap("train.bin", dtype=np.uint16, mode="r")
val = np.memmap("val.bin", dtype=np.uint16, mode="r")

print(train.shape)
print(val.shape)

print(train[:20])