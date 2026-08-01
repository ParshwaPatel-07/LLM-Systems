import regex as re
import os
from typing import BinaryIO
from collections import Counter
from collections.abc import Iterable , Iterator
import pickle


def pre_tokenization(text, special_tokens=None):

    if special_tokens is None:
        special_tokens = []
    gpt2pat = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
    if special_tokens:
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted(special_tokens,key=len,reverse=True) ) + ")"

        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    tokens = []
    for chunk in chunks:
        if chunk in special_tokens:
            tokens.append([chunk.encode("utf-8")])
            continue
        pre_tokens = re.findall(gpt2pat, chunk)

        for word in pre_tokens:
            word_token = [bytes([b]) for b in word.encode("utf-8")]
            tokens.append(word_token)

    return tokens


# print(re.findall(gpt2pat, example))

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

def build_word_frequencies(pre_tokens):

    word_freqs = Counter()

    for token in pre_tokens:
        word_freqs[tuple(token)] += 1

    return word_freqs

def FrequrncyCounting(word_frequencys):
    counts = {}

    for word,freq in word_frequencys.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts
    
def merge(pre_tokens,pair):
    
    new_ids = []
    i = 0

    while i < len(pre_tokens):
        if i < len(pre_tokens) - 1 and pre_tokens[i] == pair[0] and pre_tokens[i+1] == pair[1]:
            new_ids.append(pair[0]+pair[1])
            i += 2
        else:
            new_ids.append(pre_tokens[i])
            i += 1
    return new_ids


def train(vocab_size,tokens,special_tokens):

    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    word_freq = build_word_frequencies(tokens)
    for i in range(num_merges):
        if i % 100 == 0:
            print(f"Merge {i}/{num_merges}")
        stats = FrequrncyCounting(word_freq)
        if not stats:
            break
        pair = max(stats.items(), key=lambda x: (x[1],x[0]))[0] # choose the highest freq
        
        new_word_freq = Counter()
        for word, freq in word_freq.items():
            merged_word = tuple(merge(list(word), pair))
            new_word_freq[merged_word] += freq
        
        word_freq = new_word_freq
        merges.append(pair)

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i, (p0,p1) in enumerate(merges):
        vocab[256+i] = p0 + p1

    next_id = 256 + len(merges)
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
        
    assert len(vocab) == vocab_size

    return vocab, merges

def run_train_BPE(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str]
):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = pre_tokenization(text,special_tokens)
    vocab, meregs = train(vocab_size,tokens,special_tokens)

    return BPETokenizer(vocab,meregs,special_tokens)
    

class BPETokenizer:

    def __init__(self, 
                 vocab:dict,
                 merges:list[tuple[bytes, bytes]],
                 special_tokens:list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.byte_to_id = {v:k for k,v in vocab.items()}

    def get_pairs(self,tokens):
        return [
            (tokens[i], tokens[i + 1])
            for i in range(len(tokens) - 1)
        ]

    # def encode(self, text:str) -> list[int]:

        # pre_tokens = pre_tokenization(text, self.special_tokens)
        # final_tokens = []
       
        # for tokens in pre_tokens:
        #     # special token
            # if len(tokens) == 1 and tokens[0] in self.byte_to_id:
            #     final_tokens.extend(tokens)
            #     continue
        #     for p0, p1 in self.merges:
            #     new_tokens = []
            #     i = 0
            #     while i < len(tokens):
            #         if (i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1):
            #             new_tokens.append(p0 + p1)
            #             i += 2
            #         else:
            #             new_tokens.append(tokens[i])
            #             i += 1
            #     tokens = new_tokens
            # final_tokens.extend(tokens)

        # return [self.byte_to_id[token] for token in final_tokens]

    def encode(self, text:str) -> list[int]:
        pre_tokens = pre_tokenization(text, self.special_tokens)
        final_tokens = []
        for tokens in pre_tokens:
            if len(tokens) == 1 and tokens[0] in self.byte_to_id:
                    final_tokens.extend(tokens)
                    continue
            while True:
                    pairs = self.get_pairs(tokens)

                    best_pair = None
                    best_rank = float("inf")

                    for pair in pairs:
                        if pair in self.merge_ranks:
                            rank = self.merge_ranks[pair]
                            if rank < best_rank:
                                best_rank = rank
                                best_pair = pair
                    new_tokens = []
                    i = 0
                    if best_pair is None:
                        break
                    p0,p1 = best_pair
                    while i < len(tokens):
                            if (i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1):
                                new_tokens.append(p0 + p1)
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                    tokens = new_tokens
            final_tokens.extend(tokens)

        return [self.byte_to_id[token] for token in final_tokens]
            

    def decode(self, ids: list[int]) -> str:

        return b''.join(self.vocab[idx]for idx in ids).decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for chunk in iterable:
            yield from self.encode(chunk)
        
    @classmethod
    def from_files(cls,vocab_path:str,merge_path:str,special_tokens:list[str]):

        if special_tokens is None:
            special_tokens = []

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merge_path,"rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab=vocab,merges=merges,special_tokens=special_tokens)
    
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--vocab_out", type=str, default="vocab.pkl")
    parser.add_argument("--merges_out", type=str, default="merges.pkl")

    args = parser.parse_args()

    tokenizer = run_train_BPE(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    with open(args.vocab_out, "wb") as f:
        pickle.dump(tokenizer.vocab, f)

    with open(args.merges_out, "wb") as f:
        pickle.dump(tokenizer.merges, f)

    print("Tokenizer training complete!")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

if __name__ == "__main__":
    main()