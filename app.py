from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from models import Job
from os import environ as env
from bson import ObjectId

mongohost = env.get("MONGO_HOST", "localhost")

app = Flask(__name__)
app.config["MONGO_URI"] = f"mongodb://{mongohost}:27017/mydatabase"
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


# Rutas de Job

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
        job = Job(data['n'], data['nm'], data['t'], data['d'])
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)