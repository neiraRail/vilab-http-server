import numpy as np


class Job:
    """Representa un trabajo de medición.

    Parameters
    ----------
    node : int
        Identificador del nodo.
    num_mediciones : int
        Número de mediciones a realizar. Un valor ``-1`` indica que el
        trabajo no tiene un número máximo de mediciones (ejecución
        indefinida).
    tiempo : int
        Duración de cada medición en segundos.
    delay : int
        Tiempo de espera entre mediciones.
    active : int
        Flag que indica si el job está activo.
    ai_monitoreo : int
        Flag que indica si se ejecuta el proceso de monitoreo basado en IA
        después de cada medición.
    ai_aprendizaje : int
        Flag que indica si se ejecuta el proceso de aprendizaje basado en IA
        después de cada medición.
    """

    def __init__(self, node, num_mediciones, tiempo, delay, active,
                 ai_monitoreo, ai_aprendizaje):
        self.n = node
        self.nm = num_mediciones
        self.t = tiempo
        self.d = delay
        self.a = active
        self.ai_monitoreo = ai_monitoreo
        self.ai_aprendizaje = ai_aprendizaje

class JobRun:
    def __init__(self, job, dt):
        self.j = job["_id"]
        self.dt = dt
        self.n = job['n']
        self.nm = job['nm']
        self.d = job['d']
        self.t = job['t']

class Measure:
    """Representa una de las mediciones realizadas por un job y el análisis IA asociado"""
    def __init__(self, node, jobrun, stamp_in=None, stamp_out=None):
        self.n = node
        self.j = jobrun
        self.si = stamp_in
        self.so = stamp_out
        self.ai = None

class Marca:
    """Representa una marca que irá en la base de datos de lecturas para permitir extraer por marcas"""
    def __init__(self, timestamp, descripcion):
        self.timestamp = timestamp
        self.descripcion = descripcion


class FeatureVector:
    """Representa el vector de características asociado a una medición.

    Para almacenar ``vector`` en MongoDB se recomienda serializarlo a una
    lista mediante ``numpy.ndarray.tolist``. Esto mantiene la compatibilidad
    con BSON y permite reconstruir el arreglo original al leerlo.

        Parameters
    ----------
    measure_id : ObjectId
        Identificador de la medición a la que pertenece el vector.
    vector : np.ndarray | Iterable
        Vector de características de la medición.
    class_pred : int | None, optional
        Clase asignada por el clasificador, si existe.
    class_real : int | None, optional
        Clase real asignada manualmente. Por defecto ``None``.
    """

    def __init__(self, measure_id, vector, class_pred=None, class_real=None):
        self.m = measure_id
        self.v = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
        self.class_pred = class_pred
        self.class_real = class_real


class Clase:
    """Representa el resultado de una clasificación."""

    def __init__(self, identificador: int, descripcion: str):
        self.id = identificador
        self.descripcion = descripcion