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
