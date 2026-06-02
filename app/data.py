import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd


def load_metadata_database(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Metadata loaded successfully: {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}.")
        return None


def get_dish_metadata(df, dish_name_query):
    match = df[df["food_name_ar"].str.contains(dish_name_query, na=False, case=False)]
    if not match.empty:
        row = match.iloc[0]
        return {
            "name_ar": row["food_name_ar"],
            "name_en": row["food_name_en"],
            "ingredients_ar": row["ingredients_ar"],
            "category": row.get("main_category_ar", "Unknown")
        }
    return {
        "name_ar": dish_name_query,
        "name_en": "Unknown",
        "ingredients_ar": "Ingredients metadata not found.",
        "category": "Unknown"
    }


def build_visual_knowledge_base(model, images_dir, save_path, device, image_size=(224, 224), normalize_mean=None, normalize_std=None):
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    if os.path.exists(save_path):
        print("Loading existing visual knowledge base...")
        return torch.load(save_path, map_location=device)

    print("Building visual knowledge base with k-NN logic...")
    preprocess = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    knowledge_base = {}
    with torch.no_grad():
        for dish_name in os.listdir(images_dir):
            dish_path = os.path.join(images_dir, dish_name)
            if not os.path.isdir(dish_path):
                continue

            embeddings = []
            for img_name in os.listdir(dish_path):
                img_path = os.path.join(dish_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)

                    raw_emb = model.extract_raw_features(img_tensor)
                    emb = F.normalize(raw_emb, p=2, dim=1)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

            if embeddings:
                stacked_embeddings = torch.cat(embeddings, dim=0)
                knowledge_base[dish_name] = stacked_embeddings
                print(f"Processed '{dish_name}': {len(embeddings)} images.")

    torch.save(knowledge_base, save_path)
    print("Knowledge base saved.")
    return knowledge_base
