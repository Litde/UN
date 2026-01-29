import joblib

KMEANS_MODEL_DIR = "wikiart_kmeans_model.joblib"

def predict_cluster(features):
    kmeans = joblib.load(KMEANS_MODEL_DIR)
    new_cluster = kmeans.predict(features)
    return new_cluster

