import json
from llmembed import LLMEmbed
import numpy as np

# Initialize embedder
embedder = LLMEmbed()

# Define categories
categories = [
    "brood",
    "kaas",
    "vlees",
    "vis",
    "groente",
    "fruit",
    "zuivel",
    "beleg",
    "ontbijt",
    "dranken",
    "bier",
    "wijn",
    "koffie",
    "thee",
    "snacks",
    "zoetigheid",
    "kant-en-klaar",
    "sauzen",
    "huishouden",
    "verzorging",
    "rijst en granen"
]

# Precompute embeddings for categories
category_embeddings = embedder.encode(categories)

def categorize_text(text: str):
    text_embedding = embedder.encode([text])[0]
    similarities = np.dot(category_embeddings, text_embedding) / (
        np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(text_embedding) + 1e-10
    )
    return categories[np.argmax(similarities)]

def classify_offers(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        offers = json.load(f)

    for offer in offers:
        combined_text = f"{offer.get('product', '')} {offer.get('productInfo', '')}"
        category = categorize_text(combined_text)
        offer['category'] = category

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(offers, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]  # e.g. 'offers.json'
    output_file = sys.argv[2]  # e.g. 'offers_categorized.json'
    classify_offers(input_file, output_file)
