import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def show_similarity_matrix(texts, text_features):
    if hasattr(text_features, 'detach'):
        text_features = text_features.detach().cpu().numpy()

    sim_matrix = text_features @ text_features.T

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=texts, yticklabels=texts, cbar=False, square=True)
    plt.title("Class Similarity")
    plt.tight_layout()
    plt.show()

def show_similarity_overlay(image, sim_map, text_label):
    sim_map = np.array(sim_map)
    heatmap_resized = Image.fromarray(sim_map).resize(image.size, resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_resized)
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
    plt.title(f"Query: {text_label}")
    plt.axis('off')
    plt.show()

def show_image(image, title=None):
    if title is None:
        title=""

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"{title}")
    plt.axis('off')
    plt.show()