from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_executor import Executor
from flask_cors import CORS
from models import Job, JobRun, Measure, Marca
from os import environ as env
from bson import ObjectId
import time
from pymongo import ASCENDING

mongohost = env.get("MONGO_HOST", "localhost")

app = Flask(__name__)
CORS(app)
app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True
app.config["MONGO_URI"] = f"mongodb://{mongohost}:27017/mydatabase"

executor = Executor(app)
mongo = PyMongo(app)
mongo_inercial = PyMongo(app, uri = f'mongodb://{mongohost}:27017/inercial')

# Rutas de Nodo 

@app.route('/nodo', methods=['GET'])
def get_nodos():
    nodos = mongo.db.nodos.find({}, {'_id': 0})
    return jsonify(list(nodos))

@app.route('/init', methods=['POST'])
def init():
    if not request.is_json:
        return "Request no contiene un Json", 400
    json = request.json
    if "n" not in json:
        return "json no contiene 'n'", 400
    if "s" not in json:
        return "json no contiene 's'", 400
    
    if json["n"] == 0:
        json["n"] = 1
    nodo = mongo.db.nodos.find_one_or_404({"n": json["n"]})
    mongo.db.nodos.update_one({"_id": nodo["_id"]}, {"$set": {"s": json["s"]}})

    nodo = mongo.db.nodos.find_one_or_404({"n": json["n"]}, {"_id": 0})

    
    return jsonify(nodo)

def encender_nodo(nodo_id):
    mongo.db.nodos.find_one_and_update({"n": nodo_id}, {"$set": {"a": 1}})
    # return jsonify(nodo)

def apagar_nodo(nodo_id):
    mongo.db.nodos.find_one_and_update({"n": nodo_id}, {"$set": {"a": 0}})
    # return jsonify(nodo)

# end point para encender nodo
@app.route('/nodo/run/<nodo_id>', methods=['POST'])
def run_nodo(nodo_id):
    try:
        print("Encendiendo nodo: ", nodo_id)
        encender_nodo(int(nodo_id))
        return {'message': 'Nodo encendido exitosamente'}, 200
    except Exception as e:
        return {'error': f'Error al encender el nodo: {e}'}, 500
    
# end point para apagar nodo
@app.route('/nodo/stop/<nodo_id>', methods=['POST'])
def stop_nodo(nodo_id):
    try:
        apagar_nodo(int(nodo_id))
        return {'message': 'Nodo apagado exitosamente'}, 200
    except Exception as e:
        return {'error': f'Error al apagar el nodo: {e}'}, 500

# Rutas de Job

def run_job(job):
    # Create jobrun in mongodb
    jobrun = JobRun(job, time.time())
    jobrun_id = mongo.db.jobruns.insert_one(vars(jobrun)).inserted_id


    for i in range(job['nm']):
        # Consultar si el job sigue activo (para detectar detención por medio externo)
        active = mongo.db.jobs.find_one({"_id": job["_id"]}, {"a": 1, "_id": 0})["a"]
        if active != 1:
            break

        # Crear elemento medición en la base de datos
        measure = Measure(job['n'], jobrun_id)
        measure_id = mongo.db.measures.insert_one(vars(measure)).inserted_id


        # Ingresar marca para obtener stamp_in
        marca = Marca(time.time(), f"Inicio medición {i} del jobrun {jobrun_id}")
        stamp_in = mongo_inercial.db[f"lecturas{job['n']}"].insert_one(vars(marca)).inserted_id
        
        # Actualizar el elemento de medición con el stamp_in
        mongo.db.measures.update_one({"_id": measure_id}, {"$set": {"si": stamp_in}})
        measure.si = stamp_in

        # Loop principal del job
        encender_nodo(job['n'])
        time.sleep(job['t'])
        apagar_nodo(job['n'])
        time.sleep(job['d'])

        
        # Ingresar marca para obtener stamp_out
        marca = Marca(time.time(), f"Fin medición {i} del jobrun {jobrun_id}")
        stamp_out = mongo_inercial.db[f"lecturas{job['n']}"].insert_one(vars(marca)).inserted_id

        # Actualizar el elemento de medición con el stamp_out
        mongo.db.measures.update_one({"_id": measure_id}, {"$set": {"so": stamp_out}})
        measure.so = stamp_out

        # llamada a servicio de análisis
        if job['ai'] == 1:
            executor.submit(run_analysis, job, jobrun_id, measure, measure_id)
    
    mongo.db.jobs.update_one({"_id": job["_id"]}, {"$set": {"a": 0}})

@app.route('/job', methods=['GET'])
def get_jobs():
    cursor = mongo.db.jobs.find({})
    jobs = []
    for job in cursor:
        # Convert ObjectId to string
        job["_id"] = str(job["_id"])
        jobs.append(job)
    return jsonify(jobs)

@app.route('/job', methods=['POST'])
def create_job():
    data = request.json
    try:
        job = Job(data['n'], data['nm'], data['t'], data['d'], 0, data['ai'])
        job_id = mongo.db.jobs.insert_one(vars(job)).inserted_id
    except KeyError as e:
        return {'error': f'Falta el campo {e}'}, 400
    except BaseException as e:
        return {'error': f'Error desconocido: {e}'}, 500
    
    return {'message': 'Job creado exitosamente', 'job_id': str(job_id)}, 201

@app.route('/job/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    result = mongo.db.jobs.delete_one({'_id': ObjectId(job_id)})
    if result.deleted_count == 0:
        return {'error': 'Job no encontrado'}, 404
    return {'message': 'Job eliminado exitosamente'}, 200

@app.route('/job/run/<job_id>', methods=['POST'])
def start_job(job_id):
    job = mongo.db.jobs.find_one({'_id': ObjectId(job_id)})
    if not job:
        return {'error': 'Job no encontrado'}, 404
    
    
    mongo.db.jobs.update_one({"_id": ObjectId(job_id)}, {"$set": {"a": 1}})
    executor.submit(run_job, job)
    return {'message': 'Job iniciado exitosamente'}, 200

@app.route('/job/stop/<job_id>', methods=['POST'])
def stop_job(job_id):
    job = mongo.db.jobs.find_one({'_id': ObjectId(job_id)})
    if not job:
        return {'error': 'Job no encontrado'}, 404
    
    mongo.db.jobs.update_one({"_id": ObjectId(job_id)}, {"$set": {"a": 0}})
    return {'message': 'Job detenido exitosamente'}, 200


# get jobruns on path "/jobrun/<jobid>"
@app.route('/jobrun/<job_id>', methods=['GET'])
def get_jobruns(job_id):
    jobruns = mongo.db.jobruns.find({"j": ObjectId(job_id)})
    result = []
    for jobrun in jobruns:
        jobrun["_id"] = str(jobrun["_id"])
        jobrun["j"] = str(jobrun["j"])
        result.append(jobrun)
    return jsonify(result)

# Rutas de Eventos/lecturas

@app.route('/lecturas/node/<node>/start/<start>/pag/<pag>/<size>', methods=['GET'])
def get_lecturas(node, start, pag, size):
    # Calculate the number of documents to skip
    skip_documents = (int(pag) - 1) * int(size)

    # Create the query filter
    query = {
        'st': int(start)
    }

    # Retrieve the documents
    result = mongo_inercial.db[f'lecturas{node}'].find(query, {"_id":0}).sort("tm", ASCENDING).skip(skip_documents).limit(int(size))
    
    return jsonify(list(result))

# Rutas de Análisis

def run_analysis(job, jobrun_id, measure, measure_id):
    print("El id del job es: ", job["_id"])
    print("El id del jobrun es: ", jobrun_id)
    print("Los datos provienen del nodo: ", job["n"])
    print("Las mediciones tienen una duración de: ", job["t"])
    print(f"Se realizan mediciones cada {(job["t"] + job["d"])} segundos")
    # Obtener datos para ejecutar análisis
    datos_segmento = mongo_inercial.db[f'lecturas{job["n"]}'].find({
    "_id": {
        "$gte": measure.si,
        "$lte": measure.so
    }
    }).sort("_id", 1)

    # Ejecutar el análisis usando el servicio necesario
    count = len(list(datos_segmento))
    print("El largo de los datos del segmento es de: ", count)
    analisis = {
        "largo": count
    }
    # analisis = requests.post("")


    # Insertar análisis en la measure
    mongo.db.measures.update_one({"_id": measure_id}, {"$set": {"ai": str(analisis)}})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)