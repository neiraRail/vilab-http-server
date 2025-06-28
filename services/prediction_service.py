import numpy as np
import mlflow


def run_prediction(data: np.ndarray, model_uri: str = "models:/prediction/1") -> np.ndarray:
    """Ejecuta una predicción usando un modelo registrado en MLflow.

    Parameters
    ----------
    data : numpy.ndarray
        Datos de entrada para el modelo.
    model_uri : str, optional
        URI del modelo en el registry de MLflow.

    Returns
    -------
    numpy.ndarray
        Predicciones devueltas por el modelo.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(data)
    return np.asarray(pred)