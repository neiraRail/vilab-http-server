import numpy as np
import mlflow
from models import Clase


def run_prediction(data: np.ndarray, model_uri: str = "models:/prediction/1") -> np.ndarray:
    """Execute a prediction using an MLflow registered model.

    Parameters
    ----------
    data : numpy.ndarray
        Input data for the model.
    model_uri : str, optional
        URI of the model in the MLflow registry.

    Returns
    -------
    numpy.ndarray
        Array with the model predictions.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(data)
    return np.asarray(pred)


def run_classification(data: np.ndarray, model_uri: str = "models:/classification/1") -> Clase:
    """Classify the given data using an MLflow registered model.

    Parameters
    ----------
    data : numpy.ndarray
        Input data for the classification model.
    model_uri : str, optional
        URI of the model in the MLflow registry.

    Returns
    -------
    Clase
        Classification result.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(data)
    value = pred.squeeze().item() if isinstance(pred, np.ndarray) else pred
    return Clase(value, f"Clase {value}")


def run_anomaly_detection(data: np.ndarray, model_uri: str = "models:/anomaly/1") -> np.ndarray:
    """Detect anomalies in ``data`` using a registered model.

    Parameters
    ----------
    data : numpy.ndarray
        Input data for the anomaly detection model.
    model_uri : str, optional
        URI of the anomaly detection model in the MLflow registry.

    Returns
    -------
    numpy.ndarray
        Predicted anomaly scores or labels.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(data)
    return np.asarray(pred)