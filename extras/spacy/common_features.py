import spacy
from typing import List

en_tokenizer = spacy.load("en_core_web_sm")

input_text = "this is a sample sentence! let's see"
tokenized_text: List[str] = [
    token.text.lower() for token in en_tokenizer.tokenizer(input_text)
]

print(f"Input text: {input_text}\nTokenized text: {tokenized_text}\n")