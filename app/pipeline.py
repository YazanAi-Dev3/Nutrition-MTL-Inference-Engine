import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from app.data import get_dish_metadata


def process_input_image(image_bytes, image_size=(224, 224), normalize_mean=None, normalize_std=None):
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return preprocess(img).unsqueeze(0)
    except Exception as e:
        print(f"Image processing error: {e}")
        return None


def run_hybrid_pipeline(img_tensor, model, knowledge_base, metadata_df, device, match_threshold=0.75, k=5):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        raw_features = model.extract_raw_features(img_tensor)
        norm_features = F.normalize(raw_features, p=2, dim=1)

        metrics_pred = model.predict_metrics(raw_features)
        mass_pred = metrics_pred[0, 0].item()
        cal_pred = metrics_pred[0, 1].item()

        all_similarities = []
        for dish_name, dish_embeddings in knowledge_base.items():
            sims = F.cosine_similarity(norm_features, dish_embeddings)
            for sim in sims:
                all_similarities.append((sim.item(), dish_name))

        all_similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = all_similarities[:k]

        votes = {}
        for sim, dish_name in top_k:
            votes[dish_name] = votes.get(dish_name, 0) + 1

        best_match_folder = max(votes, key=votes.get)

        winning_sims = [sim for sim, name in top_k if name == best_match_folder]
        highest_similarity = sum(winning_sims) / len(winning_sims)

    clean_dish_name = best_match_folder.replace("تصوير من الأعلى", "").replace("من الأعلى", "").strip()

    if highest_similarity >= match_threshold:
        metadata = get_dish_metadata(metadata_df, clean_dish_name)
        confidence = highest_similarity * 100
    else:
        metadata = {
            "name_ar": "Unknown",
            "name_en": "Unknown",
            "ingredients_ar": "N/A",
            "category": "N/A"
        }
        confidence = highest_similarity * 100

    return {
        "Status": "Success",
        "Cleaned_Dish_Name": clean_dish_name,
        "Match_Confidence": f"{confidence:.2f}%",
        "Metadata_Name_AR": metadata["name_ar"],
        "Metadata_Ingredients": metadata["ingredients_ar"],
        "Estimated_Mass_g": round(mass_pred, 1),
        "Estimated_Calories_kcal": round(cal_pred, 1)
    }
