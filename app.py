from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_executor import Executor
from models import Job
from os import environ as env
from bson import ObjectId
import time

mongohost = env.get("MONGO_HOST", "localhost")

app = Flask(__name__)
app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True
app.config["MONGO_URI"] = f"mongodb://{mongohost}:27017/mydatabase"

executor = Executor(app)
mongo = PyMongo(app)

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

    nodo = mongo.db.nodos.find_one_or_404({"n": json["n"]})

    
    return jsonify(nodo)

def encender_nodo(nodo_id):
    mongo.db.nodos.find_one_and_update({"n": nodo_id}, {"$set": {"a": 1}})
    # return jsonify(nodo)

def apagar_nodo(nodo_id):
    mongo.db.nodos.find_one_and_update({"n": nodo_id}, {"$set": {"a": 0}})
    # return jsonify(nodo)


# Rutas de Job

def run_job(job):
    for _ in range(job['nm']):
        # query for "a" to check if the job is still active
        active = mongo.db.jobs.find_one({"_id": job["_id"]}, {"a": 1, "_id": 0})["a"]
        if active != 1:
            break

        encender_nodo(job['n'])
        time.sleep(job['t'])
        apagar_nodo(job['n'])
        time.sleep(job['d'])
    
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
        job = Job(data['n'], data['nm'], data['t'], data['d'], 0)
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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)