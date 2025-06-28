import numpy as np
import mlflow
from models import Clase


def run_classification(data: np.ndarray, model_uri: str = "models:/classification/1") -> Clase:
    """Clasifica los datos utilizando un modelo registrado en MLflow.

    Parameters
    ----------
    data : numpy.ndarray
        Datos de entrada para el modelo de clasificación.
    model_uri : str, optional
        URI del modelo en el registry de MLflow.

    Returns
    -------
    Clase
        Instancia que representa la clasificación obtenida.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(data)
    if isinstance(pred, np.ndarray):
        value = pred.squeeze().item()
    else:
        value = pred
    return Clase(value, f"Clase {value}")