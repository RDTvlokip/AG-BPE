# -*- coding: utf-8 -*-
"""
AG-BPE Standalone Usage Script & Web Visualizer
================================================

This script demonstrates how to load and use a pre-trained AG-BPE tokenizer
and provides a real-time web interface using Gradio to visualize its behavior.

This version has been modified to use a "longest-match" strategy directly on the
vocabulary, ignoring the BPE merge rules.
"""
import json
import regex as re
from pathlib import Path
from typing import List, Dict, Tuple
import unicodedata
import gradio as gr
import html
import math

# --- TextCleaner Class (Unchanged) ---
class TextCleaner:
    """A text cleaner for AI datasets, designed to remove invisible, abnormal, and disruptive characters."""
    UNWANTED_CHARS = {
        '\ufffd', '\u200b', '\u200c', '\u200d', '\u2060', '\u2061', '\u2063',
        '\u00a0', '\u202f', '\u2007', '\u2028', '\u2029', '\ufeff', '\ue000',
        '\uf8ff', '\ue001', '\xad', '\u180e', '\u200e', '\uFE0F',
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Cleans a given string by normalizing it, removing unwanted characters, and collapsing whitespace."""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace('’', "'").replace('‘', "'")
        text = text.replace('“', '"').replace('”', '"')
        for char in cls.UNWANTED_CHARS:
            text = text.replace(char, '')
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# --- Standalone Tokenizer Class (Logic Changed) ---
class AGBPETokenizer:
    """
    A self-contained tokenizer that loads a pre-trained model from a JSON file.
    MODIFIED: This version uses a greedy longest-match algorithm on the vocabulary,
    ignoring any BPE merge rules.
    """
    def __init__(self, vocab: Dict[str, int], merges: Dict[str, int], special_tokens: Dict[str, int]):
        """Initializes the tokenizer from loaded vocabulary and merge data."""
        self.vocab = vocab
        # self.merges is no longer used, but kept for file loading compatibility
        self.special_tokens_map = special_tokens
        self.id_to_token: Dict[int, str] = {i: s for s, i in self.vocab.items()}
        
        self.pat = re.compile(r'\s*\S+')
        
        self.unk_token_id = self.vocab.get('<unk>')
        if self.unk_token_id is None:
            # Fallback for vocabularies without <unk>
            if self.vocab:
                self.unk_token_id = next(iter(self.vocab.values()))
                print(f"Warning: '<unk>' token not found. Using first token as fallback (ID: {self.unk_token_id}).")
            else:
                 raise ValueError("The vocabulary is empty and '<unk>' token is missing.")

        self.text_cleaner = TextCleaner()

    @classmethod
    def from_file(cls, filepath: str) -> 'AGBPETokenizer':
        """Class method to conveniently load a tokenizer from a JSON file path."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: '{filepath}'")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        required_keys = ['vocab', 'merges', 'special_tokens']
        if not all(key in data for key in required_keys):
            raise ValueError("The JSON file is malformed. Missing one of: vocab, merges, special_tokens.")
        return cls(data['vocab'], data['merges'], data['special_tokens'])

    def _find_best_vocab_match(self, text_chunk: str) -> List[int]:
        """
        Tokenizes a chunk of text by greedily finding the longest possible
        substring that exists in the vocabulary.
        """
        ids = []
        i = 0
        while i < len(text_chunk):
            found_match = False
            # Search for the longest possible match from current position
            for j in range(len(text_chunk), i, -1):
                substring = text_chunk[i:j]
                if substring in self.vocab:
                    ids.append(self.vocab[substring])
                    i = j  # Move pointer to the end of the match
                    found_match = True
                    break  # Exit the inner loop to continue from the new position
            
            if not found_match:
                # If no match was found (not even a single character),
                # use the unknown token and advance by one character.
                ids.append(self.unk_token_id)
                i += 1
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes a string of text into a list of token IDs."""
        cleaned_text = self.text_cleaner.clean_text(text)
        token_ids = []
        
        if add_special_tokens and (bos_id := self.special_tokens_map.get('<bos>')) is not None:
            token_ids.append(bos_id)
        
        # Pre-tokenize the text into chunks (words and their preceding spaces)
        for chunk in self.pat.findall(cleaned_text):
            # Apply the new longest-match algorithm on each chunk
            chunk_ids = self._find_best_vocab_match(chunk)
            token_ids.extend(chunk_ids)
            
        if add_special_tokens and (eos_id := self.special_tokens_map.get('<eos>')) is not None:
            token_ids.append(eos_id)
            
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string of text."""
        special_ids_to_skip = set(self.special_tokens_map.values())
        tokens = [self.id_to_token.get(token_id, '') for token_id in token_ids if token_id not in special_ids_to_skip]
        return "".join(tokens)


# --- Gradio Web Application (Unchanged) ---

TOKENIZER_FILE = "ag_bpe_tokenizer.json"
TOKENIZER_LOADED = False
ERROR_MESSAGE = ""
tokenizer = None

try:
    if not Path(TOKENIZER_FILE).exists():
        print(f"⚠️  Warning: Tokenizer file '{TOKENIZER_FILE}' not found.")
        print("Creating a dummy tokenizer file for local testing.")
        dummy_data = {
            "vocab": {"<unk>": 0, "<bos>": 1, "<eos>": 2, " comm": 3, "ent": 4, "?": 5, "Hello": 8, " world": 9, '"comm"': 10, " comment": 11},
            "merges": {" c o m m": 1, "e n t": 2, " comment":3},
            "special_tokens": {"<unk>": 0, "<bos>": 1, "<eos>": 2}
        }
        with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=2)
        print("Dummy file created. The app will use this file.")

    print(f"🧠 Loading tokenizer from '{TOKENIZER_FILE}'...")
    tokenizer = AGBPETokenizer.from_file(TOKENIZER_FILE)
    TOKENIZER_LOADED = True
    print(f"✅ Tokenizer loaded successfully. Vocabulary size: {len(tokenizer.vocab)}")

except (FileNotFoundError, ValueError, KeyError) as e:
    ERROR_MESSAGE = str(e)
    print(f"❌ ERROR loading tokenizer: {ERROR_MESSAGE}")


def visualize_tokenization(text: str) -> Tuple[str, float, float, float]:
    """
    Takes input text, tokenizes it, calculates stats, and returns
    a styled HTML string and the statistics for display.
    """
    if not TOKENIZER_LOADED or not tokenizer:
        error_html = f"<p style='color: red; font-weight: bold;'>TOKENIZER LOADING ERROR: {ERROR_MESSAGE}</p>"
        return error_html, 0.0, 0.0, 0.0
    
    if not text:
        return "<p style='color: #888;'>Please enter some text to see the visualization...</p>", 0.0, 0.0, 0.0

    encoded_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.id_to_token.get(i, f"<unk:{i}>") for i in encoded_ids]

    # --- Calculate Statistics ---
    avg_len, std_dev, ratio = 0.0, 0.0, 0.0
    if tokens:
        token_lengths = [len(t) for t in tokens]
        avg_len = sum(token_lengths) / len(token_lengths)
        if len(token_lengths) > 1:
            variance = sum([(x - avg_len) ** 2 for x in token_lengths]) / (len(token_lengths) - 1)
            std_dev = math.sqrt(variance)
        if text:
            ratio = len(tokens) / len(text)

    # --- Generate HTML ---
    colors = ["#dbeafe", "#dcfce7", "#fee2e2", "#fef3c7", "#f3e8ff", "#d1fae5", "#e0f2fe"]
    html_output = "<div style='display: flex; flex-wrap: wrap; align-items: flex-start; font-family: sans-serif;'>"
    
    for i, token_id in enumerate(encoded_ids):
        safe_token_string = html.escape(tokens[i])
        color = colors[i % len(colors)]
        html_output += f"""
        <div style="display: inline-flex; flex-direction: column; align-items: center; margin: 4px; padding: 8px 10px; border-radius: 8px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1); box-shadow: 0 1px 3px rgba(0,0,0,0.05); text-align: center;">
            <span style="font-size: 1.1em; font-weight: 500; color: #111827; white-space: pre-wrap;">{safe_token_string}</span>
            <span style="font-size: 0.9em; font-weight: 700; color: #1e3a8a; margin-top: 5px; background-color: rgba(255,255,255,0.6); padding: 2px 6px; border-radius: 5px;">{token_id}</span>
        </div>"""
    html_output += "</div>"
    
    return html_output, round(avg_len, 2), round(std_dev, 2), round(ratio, 3)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), css="footer {display: none !important}") as demo:
    gr.Markdown(
        """
        # 👁️ Real-time Tokenizer Visualizer
        Enter text in the field below to see the tokenization happen live.
        Each colored card is a "token", with its corresponding numerical ID shown below it.
        """
    )
    
    with gr.Column():
        input_textbox = gr.Textbox(
            label="Enter your text here",
            placeholder="Type something...",
            lines=5,
            show_label=False,
        )
        
        with gr.Row():
            avg_len_box = gr.Textbox(label="Avg. Token Len", interactive=False)
            std_dev_box = gr.Textbox(label="Std. Dev Len", interactive=False)
            ratio_box = gr.Textbox(label="Tokens/Chars Ratio", interactive=False)

        output_html = gr.HTML(label="Tokens and IDs")
    
    input_textbox.input(
        fn=visualize_tokenization,
        inputs=[input_textbox],
        outputs=[output_html, avg_len_box, std_dev_box, ratio_box]
    )
    
    gr.Examples(
        examples=[
            "Artificial intelligence is fascinating.",
            'Test with "quotes" and spaces.',
            "Code like `if (x==10)` and emojis 👍🚀 are handled.",
            "Hello world! This is a test of the AG-BPE tokenizer.",
            "안녕하세요",
            "Salut comment ça va ?"
        ],
        inputs=input_textbox
    )

if __name__ == "__main__":
    demo.launch()