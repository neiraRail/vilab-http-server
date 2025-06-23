class Job:
    def __init__(self, node, num_mediciones, tiempo, delay, active, ai_active):
        self.n = node
        self.nm = num_mediciones
        self.t = tiempo
        self.d = delay
        self.a = active
        self.ai = ai_active

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
