# AG-BPE: Attention-Guided Byte-Pair Encoding
### A Novel Approach for Semantic-Aware Tokenization

---

**⚠️ This repository is currently under construction. ⚠️**

**The code, pre-trained models, and benchmark results are being finalized and will be uploaded shortly. Thank you for your patience!**

---

## 🚀 The Idea: Making BPE Smarter

Standard tokenization methods like Byte-Pair Encoding (BPE) are the foundation of modern Large Language Models (LLMs). However, they operate on a simple principle: merge the most frequent pair of tokens. This "semantically blind" approach is efficient but often creates suboptimal tokens by splitting words in ways that ignore their meaning.

**Attention-Guided BPE (AG-BPE)** is a new take on this classic algorithm. Instead of just counting, AG-BPE uses a lightweight Transformer model to "understand" the context and guide the merge process.

The result? A tokenizer that favors creating subwords that are not only frequent but also **semantically coherent**.

### ✨ Key Features
- **Morphological Awareness:** Naturally identifies and preserves meaningful parts of words (e.g., `neuro-`, `-science`, `token-iz-er`).
- **Superior Vocabulary Efficiency:** Achieves better performance with a more compact vocabulary compared to standard tokenizers.
- **Perfect Reconstruction:** Guarantees lossless conversion from text to tokens and back.
- **Drop-in Replacement:** Once trained, AG-BPE can be used just like any other BPE-based tokenizer.

## 🚧 Project Status: Under Construction

This repository will soon host the full implementation of the AG-BPE tokenizer, as described in our upcoming paper.

### What's Coming:
- [ ] **Full Python Source Code:** The complete, cleaned-up code for the `IntelligentTokenizer`, including the training and benchmarking scripts.
- [ ] **Pre-trained Model:** A ready-to-use AG-BPE tokenizer model trained on a 10MB diverse corpus.
- [ ] **Research Paper:** The full PDF of our paper, "AG-BPE: Attention-Guided Byte-Pair Encoding for Semantic-Aware Tokenization", detailing the methodology and results.
- [ ] **Jupyter Notebooks:** Tutorials and examples on how to train your own AG-BPE tokenizer and how to use the pre-trained model.
- [ ] **Benchmark Data:** The full results and scripts used to compare AG-BPE against GPT-2, BERT, and T5.

## 🛠 How It Works (A Sneak Peek)

The core innovation is a hybrid scoring mechanism for BPE merges:

```
MergeScore(pair) = Frequency(pair) + λ * AttentionScore(pair)
```

A `ContextAnalyzer` (a small Transformer) runs periodically during training to calculate the `AttentionScore`, giving a semantic boost to statistically frequent pairs. This simple but powerful idea allows AG-BPE to learn the fundamental building blocks of language.

## 📬 Stay Tuned!

Star this repository to be notified when the full project is released. We are excited to share our work with the community soon.

---

### Author
**Théo M. B. CHARLET**
- **GitHub:** [@RDTvlokip](https://github.com/RDTvlokip)
