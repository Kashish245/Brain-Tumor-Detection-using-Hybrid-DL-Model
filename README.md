# Brain-Tumor-Detection-using-Hybrid-DL-Model




from src.data_loader import get_image_paths_and_labels
from src.preprocessing import preprocess_dataset, save_processed_data

if __name__ == "__main__":
    image_paths, labels = get_image_paths_and_labels("data/raw/Training")
    print(f"Loaded {len(image_paths)} images and labels.")
    X, y, label_encoder = preprocess_dataset(image_paths, labels)
    save_processed_data(X, y, label_encoder)
    print(f"Saved features: {X.shape}, labels: {y.shape}")



