import torch
from PIL import Image

def compute_similarity(features_1, features_2):
    """
    image - text
    image - image
    text - text
    """
    return features_1 @ features_2.T

def generate_similarity_map(model, preprocess, image, text_features, patch_size=64, stride=32, device="cuda"):
    w, h = image.size
    sim_maps = []

    for text_feature in text_features:
        sim_map = []
        for y in range(0, h - patch_size + 1, stride):
            row = []
            for x in range(0, w - patch_size + 1, stride):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patch_tensor = preprocess(patch).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(patch_tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                sim = compute_similarity(feat, text_feature.unsqueeze(0)).item()
                row.append(sim)
            sim_map.append(row)
        sim_maps.append(sim_map)

    return sim_maps
