import json
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

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

def load_training_examples(json_path):
    train_examples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        categories_data = json.load(f)

    for cat in categories_data:
        cat_name = cat['name']
        keywords = cat.get('keywords', [])
        ignore_words = cat.get('ignore', [])

        # Add positive examples from keywords
        for kw in keywords:
            kw_clean = kw.strip()
            if kw_clean:
                # Add some context for embedding variety
                train_examples.append(InputExample(texts=[f"{kw_clean} special", cat_name], label=1.0))

        # Add negative examples from ignore words
        for ign in ignore_words:
            ign_clean = ign.strip()
            if ign_clean:
                train_examples.append(InputExample(texts=[ign_clean, cat_name], label=0.0))

    return train_examples

# Usage example
json_file = 'categories.json'  # path to your JSON
train_examples = load_training_examples(json_file)

# Stap 3: DataLoader met kleine batch size
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Stap 4: Definieer de loss-functie voor fine-tuning
train_loss = losses.CosineSimilarityLoss(model)

# Stap 5: Fine-tune het model
num_epochs = 4
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=10,
    output_path='./fine_tuned_all_MiniLM_L6_v2'
)
# Precompute embeddings for categories
category_embeddings = model.encode(categories, convert_to_tensor=True)

def categorize_text(text: str):
    # Embed the input text
    text_embedding = model.encode(text, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = (category_embeddings @ text_embedding) / (
        np.linalg.norm(category_embeddings.cpu(), axis=1) * np.linalg.norm(text_embedding.cpu()) + 1e-10
    )

    # Find index of best matching category
    best_idx = np.argmax(similarities.cpu().numpy())
    return categories[best_idx]

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
