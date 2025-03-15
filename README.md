# Emotion-Detection-From-Text

This project focuses on **text preprocessing and classification** using **PyTorch, Transformers, and NLP techniques**. It includes **data cleaning, tokenization, and training a deep learning model** for text classification.

## Features
- Preprocesses text data by removing stopwords, expanding contractions, and normalizing text.
- Implements **BERT WordPiece Tokenizer** for text tokenization.
- Fine-tunes a Transformer-based deep learning model for text classification.
- Utilizes **PyTorch Dataset & DataLoader** for efficient training and batching.

## Applications
This project can be applied to various real-world NLP tasks, including:
- **Sentiment Analysis:** Understanding customer feedback, social media monitoring.
- **Spam Detection:** Filtering out unwanted emails and messages.
- **Topic Classification:** Categorizing news articles, legal documents, and research papers.
- **Chatbot Training:** Improving intent recognition in conversational AI.

## Installation
Run the following command to install the required dependencies:
```bash
pip install contractions transformers tokenizers torch scikit-learn nltk pandas
```

## Usage
1. **Prepare Dataset:** Ensure your text dataset (CSV format) is available in the correct structure (`label`, `text`).
2. **Run the Notebook:** Execute `proejct2.ipynb` step by step.
3. **Train Model:** Fine-tune the Transformer model using PyTorch.
4. **Evaluate Performance:** Use accuracy, loss, and other metrics for model evaluation.

## Dataset
The project loads and processes text data from a CSV file:
```python
import pandas as pd
data = pd.read_csv("path/to/text.csv")
print(data.head())
```

## Dependencies
- PyTorch
- Transformers (Hugging Face)
- Tokenizers
- Pandas
- NLTK
- Scikit-learn

## Acknowledgments
- Hugging Face `transformers`
- PyTorch Deep Learning Framework
- NLTK for NLP utilities

