import os
from typing import List
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import mlflow
from sklearn.decomposition import PCA


def _load_jobrun_data(jobrun_id: str, mongo_host: str) -> np.ndarray:
    """Load all inertial data segments for a JobRun.

    Parameters
    ----------
    jobrun_id: str
        Identifier of the JobRun document in MongoDB.
    mongo_host: str
        Hostname for the MongoDB instance.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 6)`` with the concatenated sensor readings.
    """
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


def main(jobrun_id: str, mongo_host: str = "localhost") -> None:
    """Train a PCA model on data from a JobRun and register it in MLflow."""
    data = _load_jobrun_data(jobrun_id, mongo_host)
    if data.size == 0:
        raise RuntimeError("No se encontraron datos para el JobRun especificado")

    scaler = mlflow.sklearn.load_model("models:/signal_scaler/1")
    scaled = scaler.transform(data)

    pca = PCA(n_components=0.95, random_state=0)
    pca.fit(scaled)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            pca,
            artifact_path="model",
            registered_model_name="feature_pca",
        )
        mlflow.log_param("jobrun_id", jobrun_id)
        mlflow.log_param("n_components", pca.n_components_)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrena y registra un PCA usando los datos de un JobRun"
    )
    parser.add_argument("jobrun_id", help="ID del JobRun en MongoDB")
    parser.add_argument(
        "--mongo-host",
        default=os.environ.get("MONGO_HOST", "localhost"),
        help="Host de MongoDB",
    )
    args = parser.parse_args()

    main(args.jobrun_id, args.mongo_host)
