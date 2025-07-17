from typing import List
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import pandas as pd
import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from .feature_engineering import (
    extract_time_features,
    extract_freq_features,
    extract_envelope_features,
)

from reusable.modelos import CNN_Autoencoder
from reusable.datasets import TimeSeriesDataset
from reusable.train_test_and_save import train_ae, EarlyStopping

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



def train_and_save_anomaly_detector(
    jobrun_id: str, mongo_host: str = "localhost"
) -> None:
    """Train an IsolationForest model on PCA features and log it to MLflow."""
    data = _load_jobrun_data(jobrun_id, mongo_host)
    if data.size == 0:
        raise RuntimeError("No se encontraron datos para el JobRun especificado")

    scaler = mlflow.sklearn.load_model("models:/signal_scaler/1")
    scaled = scaler.transform(data)
    features = _compute_features(scaled)
    if features.size == 0:
        raise RuntimeError("No hay suficientes ventanas de datos para entrenar el modelo")

    pca = mlflow.sklearn.load_model("models:/feature_pca/1")
    projected = pca.transform(features)

    detector = IsolationForest(random_state=0)
    detector.fit(projected)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            detector,
            artifact_path="model",
            registered_model_name="anomaly",
        )
        mlflow.log_param("jobrun_id", jobrun_id)


def train_and_save_predictor(
    jobrun_id: str, mongo_host: str = "localhost"
) -> None:
    """Train a simple feature predictor and log it to MLflow."""
    data = _load_jobrun_data(jobrun_id, mongo_host)
    if data.size == 0:
        raise RuntimeError("No se encontraron datos para el JobRun especificado")

    scaler = mlflow.sklearn.load_model("models:/signal_scaler/1")
    scaled = scaler.transform(data)
    features = _compute_features(scaled)
    if features.shape[0] < 2:
        raise RuntimeError("Se requieren al menos dos ventanas para entrenar el predictor")

    pca = mlflow.sklearn.load_model("models:/feature_pca/1")
    projected = pca.transform(features)

    X = projected[:-1]
    y = projected[1:]

    model = LinearRegression()
    model.fit(X, y)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="prediction",
        )
        mlflow.log_param("jobrun_id", jobrun_id)



def train_and_save_cnnae(
    jobrun_id: str,
    mongo_host: str = "localhost",
    latent_size: int = 32,
    epochs: int = 10,
    batch_size: int = 32,
) -> None:
    """Train a CNN Autoencoder on JobRun windows and log it to MLflow.

    Parameters
    ----------
    jobrun_id : str
        Identifier of the JobRun used for training.
    mongo_host : str, optional
        MongoDB host where the inertial data is stored.
    latent_size : int, optional
        Dimension of the latent space of the autoencoder.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size used for training.
    """

    data = _load_jobrun_data(jobrun_id, mongo_host)
    if data.size == 0:
        raise RuntimeError("No se encontraron datos para el JobRun especificado")

    scaler = mlflow.sklearn.load_model("models:/signal_scaler/1")
    scaled = scaler.transform(data)

    df = pd.DataFrame(
        scaled, columns=["ax", "ay", "az", "gx", "gy", "gz"]
    )

    dataset = TimeSeriesDataset(df, WINDOW_SIZE)
    if len(dataset) == 0:
        raise RuntimeError(
            "No hay suficientes ventanas de datos para entrenar el autoencoder"
        )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Autoencoder(latent_size).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    for epoch in range(epochs):
        train_loss, stop = train_ae(
            model,
            train_loader,
            loss_fn,
            optimizer,
            early_stopping=early_stopping,
            device=device,
        )

        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                X = batch.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, X).item()
        test_loss /= len(test_loader)

        if stop:
            break

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="anomaly",
        )
        mlflow.log_param("jobrun_id", jobrun_id)
        mlflow.log_param("latent_size", latent_size)
        mlflow.log_param("epochs", epoch + 1)
        mlflow.log_metric("final_train_loss", float(train_loss))
        mlflow.log_metric("final_test_loss", float(test_loss))
