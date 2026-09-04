# clip_prompt.py - File used to create prompt db for clip
import os
import torch
import clip
import numpy as np
from config import CLIP_TEXT_EMBEDDINGS_PATH

PROMPTS = {
    "amazon_delivery": "a person wearing an Amazon delivery uniform",
    "fedex_delivery": "a person wearing a FedEx uniform",
    "ups_delivery": "a person wearing a UPS uniform",
    "food_delivery": "a person carrying an insulated food delivery bag",
    "food_delivery_branded": "a person wearing DoorDash or Uber Eats branded gear",
    "regular_visitor": "a person with no delivery indicators, a regular visitor",
}

def main():
    device = "cpu"
    print("Loading CLIP (PyTorch, CPU, one-time)...")
    # Load the clip ViT-B/32 model
    model, _ = clip.load("ViT-B/32", device=device)

    # sets the model in infer or evaluation mode (explicitly comes out of training mode)
    model.eval() # for avoiding drop outs at different layers

    # Creates the list of labels from prompts
    labels = list(PROMPTS.keys())
    # 1. Creates a list of prompt value strings in the same order as labels
    # 2. Tokenizes into integer values 
    # 3. then writes it to the device memory (cpu here)
    texts = clip.tokenize([PROMPTS[l] for l in labels]).to(device)

    # no_grad -- implies inference (not training) - for memory computing optimization
    with torch.no_grad():
        # Actual forward Pass
        # Feeds the tokenized integer IDs through CLIP's text transformer producing the raw output embeddings
        # Shape -- (6, 512) -- 512 vector per prompt
        text_features = model.encode_text(texts) # Actual forward pass
        # L2 normalization step
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # numpy floating point conversion
    embeddings = text_features.cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(CLIP_TEXT_EMBEDDINGS_PATH), exist_ok=True)
    np.savez(CLIP_TEXT_EMBEDDINGS_PATH, labels=labels, embeddings=embeddings)
    print(f"Saved {len(labels)} text embeddings to {CLIP_TEXT_EMBEDDINGS_PATH}")
    for l in labels:
        print(f"  - {l}: \"{PROMPTS[l]}\"")


if __name__ == "__main__":
    main()
