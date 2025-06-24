import pandas as pd
import numpy as np


def generar_feature_vector(datos_segmento):
    """Calcula un vector de características básico a partir de datos de medición.

    Parameters
    ----------
    datos_segmento : Iterable[dict]
        Datos de lecturas provenientes de MongoDB.

    Returns
    -------
    numpy.ndarray
        Vector de características calculado.
    """
    df = pd.DataFrame(list(datos_segmento))
    if '_id' in df:
        df.drop(columns=['_id'], inplace=True)
    if df.empty:
        return np.array([])
    # Ejemplo simple: promedio de cada columna
    return df.mean().to_numpy()
