import numpy as np
from pathlib import Path
from tqdm import tqdm

from tokenizer import BPETokenizer

# ==========================================================
# Paths
# ==========================================================

# TRAIN_TXT = Path("/content/drive/MyDrive/Transformer AS-1/data/owt_train.txt")
# VAL_TXT   = Path("/content/drive/MyDrive/Transformer AS-1/data/owt_val.txt")

# TRAIN_BIN = Path("/content/drive/MyDrive/Transformer AS-1/data/owt_train.bin")
# VAL_BIN   = Path("/content/drive/MyDrive/Transformer AS-1/data/owt_val.bin")

# VOCAB = "/content/drive/MyDrive/Transformer AS-1/tokenizer/vocab.pkl"
# MERGES = "/content/drive/MyDrive/Transformer AS-1/tokenizer/merges.pkl"

from pathlib import Path

ROOT = Path(__file__).resolve().parent               # cs336_basics
ASSIGNMENTS = ROOT.parent.parent                     # Assignments
DATA = ASSIGNMENTS / "data"

TRAIN_TXT = DATA / "owt_train.txt"
VAL_TXT   = DATA / "owt_valid.txt"

TRAIN_BIN = DATA / "owt_train.bin"
VAL_BIN   = DATA / "owt_valid.bin"

VOCAB = ROOT / "vocab.pkl"
MERGES = ROOT / "merges.pkl"

# ==========================================================
# Tokenizer
# ==========================================================

print("Loading tokenizer...")

tokenizer = BPETokenizer.from_files(
    vocab_path=VOCAB,
    merge_path=MERGES,
    special_tokens=["<|endoftext|>"],
)

print("Tokenizer loaded.\n")

# ==========================================================
# Preprocess
# ==========================================================

def preprocess(input_path: Path, output_path: Path):

    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "wb") as fout:

        tokens = []

        for token in tqdm(
            tokenizer.encode_iterable(fin),
            desc=input_path.name,
        ):
            tokens.append(token)

            # write every million tokens
            if len(tokens) >= 1_000_000:
                np.asarray(tokens, dtype=np.uint16).tofile(fout)
                total_tokens += len(tokens)
                tokens.clear()

        if tokens:
            np.asarray(tokens, dtype=np.uint16).tofile(fout)
            total_tokens += len(tokens)

    print(f"\nFinished.")
    print(f"Tokens written : {total_tokens:,}")
    print()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    # preprocess(TRAIN_TXT, TRAIN_BIN)
    preprocess(VAL_TXT, VAL_BIN)

    print("=" * 70)
    print("Finished preprocessing OpenWebText.")
    print("=" * 70)