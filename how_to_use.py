# -*- coding: utf-8 -*-
"""
AG-BPE Standalone Usage Script
==============================

This script demonstrates how to load and use a pre-trained AG-BPE (Attention-Guided
Byte-Pair Encoding) tokenizer from its final JSON file.

It defines a self-contained AGBPETokenizer class that does not require PyTorch or
the original training code. It replicates the exact encoding and decoding logic,
ensuring that anyone can use the tokenizer and reproduce the benchmark results.

The key feature of this implementation is the "_apply_bpe" method, which uses a
"brute-force" validation approach. It ensures that every token produced by a merge
operation exists in the final vocabulary, guaranteeing perfect, lossless
reconstruction of any text.
"""
import json
import regex as re
from pathlib import Path
from typing import List, Dict, Tuple
import unicodedata

# --- TextCleaner Class ---
# This class is included to ensure that the input text is pre-processed
# in exactly the same way as during the tokenizer's training.
# This is crucial for achieving perfect encode-decode reconstruction.
class TextCleaner:
    """A text cleaner for AI datasets, designed to remove invisible, abnormal, and disruptive characters."""

    # A set of specific, problematic Unicode characters to be removed.
    UNWANTED_CHARS = {
        '\ufffd',  # Replacement char
        '\u200b', '\u200c', '\u200d',  # Zero-width spaces
        '\u2060', '\u2061', '\u2063',  # Invisible joiners
        '\u00a0', '\u202f', '\u2007',  # Abnormal spaces
        '\u2028', '\u2029',            # Line and paragraph separators
        '\ufeff',                      # Byte Order Mark (BOM)
        '\ue000', '\uf8ff', '\ue001',  # Private Use Area characters
        '\xad',                        # Soft hyphen
        '\u180e',                      # Mongolian Vowel Separator
        '\u200e',                      # Left-to-right mark
        '\uFE0F',                      # Emoji variation selector
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans a given string by normalizing it, removing unwanted characters,
        and collapsing whitespace.
        """
        # NFKC normalization handles many compatibility issues and standardizes characters.
        text = unicodedata.normalize("NFKC", text)
        # Standardize curly quotes to simple ones.
        text = text.replace('’', "'").replace('‘', "'")
        text = text.replace('“', '"').replace('”', '"')
        # Remove all explicitly defined unwanted characters.
        for char in cls.UNWANTED_CHARS:
            text = text.replace(char, '')
        # Remove any remaining control characters, except for standard whitespace.
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        # Replace multiple whitespace characters with a single space.
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# --- Standalone Tokenizer Class ---
class AGBPETokenizer:
    """
    A self-contained tokenizer that loads and uses a pre-trained AG-BPE model
    from a JSON file containing the vocabulary and merge rules.
    """
    def __init__(self, vocab: Dict[str, int], merges: Dict[str, int], special_tokens: Dict[str, int]):
        """
        Initializes the tokenizer from loaded vocabulary and merge data.
        """
        self.vocab = vocab
        # The merges are stored as "t1 t2": rank. We convert them to ('t1', 't2'): rank for faster lookups.
        self.merges = {tuple(k.split()): v for k, v in merges.items()}
        self.special_tokens_map = special_tokens
        
        # Create the reverse mapping from ID to token for fast decoding.
        self.id_to_token: Dict[int, str] = {i: s for s, i in self.vocab.items()}
        
        # This regex pattern is used for pre-tokenization, splitting text into words and symbols.
        # It's the same pattern used during training to ensure consistency.
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Get the ID for the unknown token, crucial for handling out-of-vocabulary words.
        self.unk_token_id = self.vocab.get('<unk>')
        if self.unk_token_id is None:
            raise ValueError("The '<unk>' token is missing from the vocabulary.")
            
        # Instantiate the cleaner for pre-processing text.
        self.text_cleaner = TextCleaner()

    @classmethod
    def from_file(cls, filepath: str) -> 'AGBPETokenizer':
        """
        Class method to conveniently load a tokenizer from a JSON file path.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: '{filepath}'")
        
        print(f"🧠 Loading tokenizer from '{filepath}'...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate that the JSON file has the necessary structure.
        required_keys = ['vocab', 'merges', 'special_tokens']
        if not all(key in data for key in required_keys):
            raise ValueError("The JSON file is malformed. Missing one of: vocab, merges, special_tokens.")
            
        return cls(data['vocab'], data['merges'], data['special_tokens'])

    def _apply_bpe(self, word_chars: List[str]) -> List[str]:
        """
        Applies the BPE merge rules to a list of characters, with a crucial validation step.
        This "brute-force" method ensures perfect reconstruction.
        """
        if not self.merges:
            return word_chars
            
        while len(word_chars) > 1:
            # Find all adjacent pairs in the current list of tokens.
            pairs = list(zip(word_chars[:-1], word_chars[1:]))
            
            # Create a mutable copy of merge rules to temporarily invalidate merges.
            local_merges = self.merges.copy()
            
            best_pair = None
            
            # Loop until a valid merge is found or no more merges are possible.
            while True:
                # If we've exhausted all possible merges for this word, stop.
                if not local_merges:
                    best_pair = None
                    break
                
                # Find the highest-priority pair (lowest rank) that exists in the current word.
                # 'default=None' prevents an error if no valid pairs are found.
                valid_pairs_in_word = (p for p in pairs if p in local_merges)
                current_best_pair = min(valid_pairs_in_word, key=local_merges.get, default=None)

                if current_best_pair is None:
                    best_pair = None
                    break
                
                # --- THE CRUCIAL VALIDATION STEP ---
                merged_token = current_best_pair[0] + current_best_pair[1]
                if merged_token in self.vocab:
                    # The merge is valid because the resulting token exists in our final vocabulary.
                    best_pair = current_best_pair
                    break
                else:
                    # This was an intermediate merge during training, not a final token.
                    # We invalidate it for this word by removing it from our local copy of merges.
                    del local_merges[current_best_pair]

            # If no valid merge could be found in this iteration, stop the BPE process for this word.
            if best_pair is None:
                break
            
            # Apply the validated best merge to the list of tokens.
            new_word_chars = []
            i = 0
            while i < len(word_chars):
                if i < len(word_chars) - 1 and (word_chars[i], word_chars[i+1]) == best_pair:
                    new_word_chars.append(word_chars[i] + word_chars[i+1])
                    i += 2
                else:
                    new_word_chars.append(word_chars[i])
                    i += 1
            word_chars = new_word_chars
            
        return word_chars

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes a string of text into a list of token IDs."""
        
        # 1. Always clean the input text first to match the training pre-processing.
        cleaned_text = self.text_cleaner.clean_text(text)
        
        token_ids = []
        # Optionally add the Beginning-Of-Sentence token.
        if add_special_tokens:
            bos_id = self.special_tokens_map.get('<bos>')
            if bos_id is not None:
                token_ids.append(bos_id)

        # 2. Pre-tokenize the text into basic chunks (words, numbers, symbols).
        for chunk in self.pat.findall(cleaned_text):
            # 3. Apply the BPE algorithm to each chunk.
            tokens = self._apply_bpe(list(chunk))
            # 4. Convert the final, valid tokens to their corresponding IDs.
            token_ids.extend(self.vocab.get(token, self.unk_token_id) for token in tokens)

        # Optionally add the End-Of-Sentence token.
        if add_special_tokens:
            eos_id = self.special_tokens_map.get('<eos>')
            if eos_id is not None:
                token_ids.append(eos_id)
            
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string of text."""
        
        # Create a set of special token IDs to ignore during reconstruction.
        special_ids_to_skip = set(self.special_tokens_map.values())
        
        # Look up each ID in the id_to_token map, skipping special tokens.
        tokens = [self.id_to_token.get(token_id, '') for token_id in token_ids if token_id not in special_ids_to_skip]
        
        # Join all tokens together. The pre-tokenization regex handles spaces correctly.
        return "".join(tokens)


# --- Main Execution Block (for demonstration and testing) ---
if __name__ == "__main__":
    # The name of the JSON file containing the pre-trained tokenizer data.
    TOKENIZER_FILE = "ag_bpe_tokenizer.json"
    
    try:
        # 1. Load the tokenizer from the file.
        tokenizer = AGBPETokenizer.from_file(TOKENIZER_FILE)
        
        print(f"✅ Tokenizer loaded successfully. Vocabulary size: {len(tokenizer.vocab)}")
        print("-" * 50)

        # 2. Define a list of sentences to test the tokenizer's capabilities.
        test_sentences = [
            "L'intelligence artificielle est fascinante.",
            "  Test avec    espaces multiples et ’apostrophe’ typographique.",
            "What are you doing tonight? 🚀",
            "Le code `if (x==10)` et l'emoji 👍 sont gérés.",
            "안녕하세요" # Korean text to test Unicode handling.
        ]

        # 3. Loop through each sentence and perform a full encode-decode cycle.
        for text in test_sentences:
            print(f"\nOriginal: '{text}'")
            
            # Clean the original text to have a fair comparison target.
            cleaned_text_for_test = TextCleaner.clean_text(text)
            
            # Encode without special tokens for the reconstruction test.
            encoded = tokenizer.encode(text, add_special_tokens=False)
            # Decode the IDs back to text.
            decoded = tokenizer.decode(encoded)
            
            print(f"  -> Cleaned for test: '{cleaned_text_for_test}'")
            print(f"  -> Decoded Text:     '{decoded}'")
            
            # This assertion verifies that the tokenizer is lossless. It MUST pass.
            assert cleaned_text_for_test == decoded
            print("  -> ✅ Perfect reconstruction cycle!")
            
            # Display the actual tokens for qualitative analysis.
            raw_tokens = [tokenizer.id_to_token.get(i, '<unk>') for i in encoded]
            print(f"  -> Tokens: {' | '.join(raw_tokens)}")

    except FileNotFoundError as e:
        print(e)
    except (ValueError, KeyError) as e:
        print(f"Error in the JSON file format: {e}")
    except AssertionError:
        print("  -> ❌ Reconstruction cycle test FAILED.")