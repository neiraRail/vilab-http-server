from typing import List
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .feature_engineering import (
    extract_time_features,
    extract_freq_features,
    extract_envelope_features,
)

WINDOW_SIZE = 100


def _load_jobrun_data(jobrun_id: str, mongo_host: str) -> np.ndarray:
    """Load all inertial data segments for a JobRun."""
    client = MongoClient(mongo_host, 27017)
    db = client["mydatabase"]
    inertial_db = client["inercial"]

    measures = list(db.measures.find({"j": ObjectId(jobrun_id)}))
    segments: List[np.ndarray] = []
    for measure in measures:
        si = measure.get("si")
        so = measure.get("so")
        node = measure.get("n")
        if not si or not so or node is None:
            continue
        query = {"_id": {"$gte": si, "$lte": so}}
        cursor = inertial_db[f"lecturas{node}"].find(query).sort("_id", 1)
        data = np.array([
            [
                doc.get("ax", 0.0),
                doc.get("ay", 0.0),
                doc.get("az", 0.0),
                doc.get("gx", 0.0),
                doc.get("gy", 0.0),
                doc.get("gz", 0.0),
            ]
            for doc in cursor
        ], dtype=float)
        if data.size:
            segments.append(data)

    if not segments:
        return np.empty((0, 6), dtype=float)
    return np.vstack(segments)


def _compute_features(data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Compute normalized feature vectors for each window in ``data``."""
    if data.shape[0] < window_size:
        return np.empty((0, 0), dtype=float)

    feats: list[np.ndarray] = []
    for start in range(0, data.shape[0] - window_size + 1, window_size):
        window = data[start : start + window_size]
        time_f = extract_time_features(window)
        freq_f = extract_freq_features(window)
        env_f = extract_envelope_features(window)
        full = np.concatenate([time_f, freq_f, env_f])
        feats.append(full)

    return np.vstack(feats)


def train_and_save_scaler(jobrun_id: str, mongo_host: str = "localhost") -> None:
    """Train a ``StandardScaler`` on JobRun data and log it to MLflow."""
    data = _load_jobrun_data(jobrun_id, mongo_host)
    if data.size == 0:
        raise RuntimeError("No se encontraron datos para el JobRun especificado")

    scaler = StandardScaler()
    scaler.fit(data)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            scaler,
            artifact_path="model",
            registered_model_name="signal_scaler",
        )
        mlflow.log_param("jobrun_id", jobrun_id)


def train_and_save_pca(jobrun_id: str, mongo_host: str = "localhost") -> None:
    """Train a PCA model on JobRun data and log it to MLflow."""
    data = _load_jobrun_data(jobrun_id, mongo_host)
    if data.size == 0:
        raise RuntimeError("No se encontraron datos para el JobRun especificado")

    scaler = mlflow.sklearn.load_model("models:/signal_scaler/1")
    scaled = scaler.transform(data)

    feature_matrix = _compute_features(scaled)
    if feature_matrix.size == 0:
        raise RuntimeError("No hay suficientes ventanas de datos para entrenar el PCA")

    pca = PCA(n_components=128, random_state=0)
    pca.fit(feature_matrix)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            pca,
            artifact_path="model",
            registered_model_name="feature_pca",
        )
        mlflow.log_param("jobrun_id", jobrun_id)
        mlflow.log_param("n_components", pca.n_components_)
