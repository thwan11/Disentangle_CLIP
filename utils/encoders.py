import torch
import clip
from PIL import Image

def load_clip(backbone="RN101", device="cuda"):
    model, preprocess = clip.load(backbone, device=device)
    return model, preprocess

def encode_text(model, texts, device="cuda"):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

def encode_image(model, images, preprocess, device="cuda"):
    image_inputs = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        features = model.encode_image(image_inputs)
        return features / features.norm(dim=-1, keepdim=True)