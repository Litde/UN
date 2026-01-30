import joblib

KMEANS_MODEL_DIR = "clustering/wikiart_kmeans_model.joblib"

def predict_cluster(features):
    pooled = features.mean(dim=(2, 3))  # [B, C]
    kmeans = joblib.load(KMEANS_MODEL_DIR)
    new_clusters = kmeans.predict(pooled.numpy())
    return new_clusters[0]

