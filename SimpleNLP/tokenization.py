import regex
import json
from collections import defaultdict, Counter


# Word & Character Tokenization
class Tokenizer:
    def __init__(self, mode="word"):
        assert mode in ["word", "char"], "mode should be 'word' or 'char'"
        self.mode = mode

    def tokenize(self, text):
        # Word Tokenization
        if self.mode == "word":
            # Regular expression to match words, numbers with commas, contractions, and hyphenated phrases
            pattern = r"\b(?:\w+(?:[-,']\w+)*|\d{1,5}(?:,\d{3})*)\b"
            return regex.findall(pattern, text.lower())
        # Char Tokenization
        else:
            return list(text.replace(" ", ""))


# N-Gram Tokenization
class NGramTokenizer:
    def __init__(self, n=2):
        self.n = n

    def tokenize(self, text):
        words = text.split()
        return [' '.join(words[i:i+self.n]) for i in range(len(words) - self.n + 1)]


# Byte-Pair Encoding (BPE)
class BPETokenizer:
    def __init__(self, vocab_size=1000):
        """
        Initializes BPE Tokenizer.
        vocab_size: number of merge operations (subword vocabulary size).
        By default vocab_size = 1000
        """
        self.vocab_size = vocab_size
        self.bpe_ranks = {}
        self.merges = []
        self.vocab = {}

    def pre_tokenization(self, text):
        """
        Pre-tokenize text by handling spaces and splitting words correctly.
        Converts spaces into a special token: 'Ś'
        """
        text = regex.sub(r"(\s+)", r" \1", text)  # preserve spaces
        text = text.strip()
        return text.split()

    def build_vocab(self, text):
        """
        Creates an initial vocabulary where each word is split into individual characters.
        """
        word_freqs = Counter(self.pre_tokenization(text))

        # Convert each word into space-separated characters (BPE Style)
        vocab = {tuple(word) + ('<\w>', ): freq for word, freq in word_freqs.items()}  # End-of-word token
        return vocab

    def get_stats(self, vocab):
        """
        Computes the frequency of adjacent symbol pairs in the vocabulary.
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        Merge the most frequent pair in the vocabulary.
        """
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in vocab.items():
            new_word = tuple(' '.join(word).replace(bigram, replacement).split(' '))
            new_vocab[new_word] = freq

        return new_vocab

    def train(self, text):
        """
        Trains BPE on given text and stores merge operations.
        """
        vocab = self.build_vocab(text)

        for i in range(self.vocab_size):
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = i

        # create final vocabulary
        self.vocab = {"".join(word): idx for idx, word in enumerate(vocab.keys())}

    def bpe_encode(self, word):
        """
        Encodes a word using trained BPE merges.
        """
        word = tuple(word) + ('</w>',)
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            min_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if min_pair not in self.bpe_ranks:
                break
            word = tuple(
                ''.join(word[i:i + 2]) if (word[i], word[i + 1]) == min_pair else word[i] for i in range(len(word) - 1))

        return list(word)

    def tokenize(self, text):
        """
        Tokenizes input text using trained BPE.
        """
        words = self.pre_tokenization(text)
        return [self.bpe_encode(word) for word in words]

    def save_vocab(self, file_path):
        """
        Saves the trained vocabulary.
        """
        with open(file_path, "w") as f:
            json.dump(self.vocab, f)

    def load_vocab(self, file_path):
        """
        Loads a saved vocabulary.
        """
        with open(file_path, "r") as f:
            self.vocab = json.load(f)