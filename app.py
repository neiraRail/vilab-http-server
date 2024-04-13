from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from os import environ as env

mongohost = env.get("MONGO_HOST", "localhost")

app = Flask(__name__)
app.config["MONGO_URI"] = f"mongodb://{mongohost}:27017/mydatabase"
mongo = PyMongo(app)

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)