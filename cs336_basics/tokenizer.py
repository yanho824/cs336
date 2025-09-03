from ast import main
from collections import defaultdict
from email import errors
from math import e
from re import I
import regex as re
from typing import BinaryIO, Iterable, Iterator
import os
from multiprocessing import Process, Queue
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

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """split text by special_tokens."""
    if not special_tokens:
        return text

    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    pattern = '(' + '|'.join(map(re.escape, special_tokens_sorted)) + ')'

    # split
    return re.split(pattern, text)

def pretokenize(text: str, special_tokens: list[str]) -> list[bytes]:
    """Use GPT2 pattern to pretokenize"""

    # Split the text by special tokens
    parts = split_by_special_tokens(text, special_tokens)

    tokens_list = []

    # Process each part
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        if part in special_tokens:
            special_token_byte = part.encode("utf-8")
            tokens_list.append(special_token_byte)
        else:
            # Find all substrings matching the pattern
            strs_token = re.findall(PAT, part)
            tokens_list.extend(token.encode("utf-8") for token in strs_token)
    return tokens_list

def Worker(chunk, special_tokens, q):
    # Process each chunk, put the result in queue
    result = pretokenize(chunk, special_tokens)
    q.put(result)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte pair encoding (BPE) model."""
    vocab: dict[int, bytes] = dict()
    merges: list[tuple[bytes, bytes]] = []

    # merge times
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Vocab initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens):
        vocab[i + 256] = special_token.encode("utf-8")

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        # boundaries is the list of each chunk's start position
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    # parallize processing these chuncks
    q = Queue()
    processes = []
    for chunk in chunk_list:
        p = Process(target=Worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    pretokens_list = [q.get() for _ in processes]
    for p in processes:
        p.join()
    
    # Store the all tokens in text
    pretokens = [token for tokens in pretokens_list for token in tokens]
   

    # Merge the results
    for i in range(num_merges):
        # Count token pairs
        count = defaultdict(int)    
        for index, token in enumerate(pretokens):
            if index < len(pretokens) - 1:
                count[token, pretokens[index + 1]] += 1
        
        if not count: # No more pairs to merge
            break

        # Prefer lexicographically greater pair
        # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
        max_pair = max(
            count.items(),
            key = lambda x: (
                x[1], # frequecy
                x[0][0],
                x[0][1]
            )
        )[0]
    
        # Create new token
        new_index = new_index = 256 + len(special_tokens) + i
        token1, token2 = max_pair
        vocab[new_index] = token1 + token2
        merges.append((token1, token2))

        # merge in pretokens
        new_pretokens = []
        i = 0
        while i < len(pretokens):
            if i < len(pretokens) - 1 and (pretokens[i], pretokens[i + 1]) == max_pair:
                new_pretokens.append(vocab[new_index])
                i += 2  # Skip the next token as it's merged
            else:
                new_pretokens.append(pretokens[i])
                i += 1
        pretokens = new_pretokens
        
    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges"""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        pretokens = pretokenize(text, self.special_tokens) # list[bytes]
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens] # list[bytes]

        token_ids = []

        vocab_reversed = {v: k for k, v in self.vocab.items()} # {bytes: int}
        # Convert bytes to token_id
        for token in pretokens:
            if token in byte_special_tokens:
                token_ids.append(vocab_reversed[token])
            else:
                for b in token:
                    token_ids.append(vocab_reversed[bytes([b])]) 

         # Merge - apply all merges iteratively until no more merges can be applied
        for merge in self.merges:
            token1, token2 = merge
            new_token_ids = []
            index = 0
            
            while index < len(token_ids):
                if (index < len(token_ids) - 1 and 
                    self.vocab[token_ids[index]] == token1 and 
                    self.vocab[token_ids[index + 1]] == token2):
                    # Found a pair to merge
                    new_token = token1 + token2
                    new_token_ids.append(vocab_reversed[new_token])
                    index += 2  # Skip both tokens as they're merged
                else:
                    # No merge, keep the current token
                    new_token_ids.append(token_ids[index])
                    index += 1
            
            token_ids = new_token_ids

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            for idx in self.encode(line):
                yield idx

    def decode(self, token_ids: list[int]) -> str:
        byte_tokens = [self.vocab[token_id] for token_id in token_ids]
        token_str = ""
        for byte_token in byte_tokens:
            token_str += byte_token.decode("utf-8", errors="replace")
        return token_str

def main():
    file_path = r"D:\study\cs336\assignment1-basics\tests\fixtures\corpus.en"
    vocab_size = 500
    # special_tokens = ["<|endoftext|>"]
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]

    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    # print(merges)

    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    encoded = tokenizer.encode(test_string)
    print("encoded:",encoded)
    decoded = [tokenizer.decode([x]) for x in encoded]
    print("decoded:", decoded)

    print(test_string == decoded)

def test():
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|><|endoftext|>", "<|endoftext|>"})
    decoded = [tokenizer.decode([x]) for x in ids]
    print(decoded)

if __name__ == "__main__":
    main()