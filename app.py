from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import certifi
import joblib
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from bson import ObjectId
# Load environment variables
load_dotenv()
print("Loaded MONGO_URI:", os.getenv("MONGO_URI"))


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Connect to MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
patients_collection = None
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        print("Testing MongoDB connection...")
        print(client.server_info())  # forces a connection
        print("✅ MongoDB connected successfully!")
        db = client["cancerDB"]
        patients_collection = db["diagnosis"]
    except Exception as e:
        # don't crash the app on DB connection failure; warn and continue
        print(f"Warning: could not connect to MongoDB: {e}")

# Load trained model
# Resolve path relative to project root (one level above backend/)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "cancer_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}. Please place cancer_model.pkl in the project root.")
model = joblib.load(model_path)

# === ROUTES ===

@app.route("/")
def home():
    return jsonify({"message": "Cancer Classification API is running successfully!"})

# 1️⃣ Route to add a new patient and get prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data) 
        # Extract feature values (excluding name, age)
        feature_keys = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
            'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
            'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
            'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
            'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
            'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry','worst_fractal_dimension'
        ]

        # Convert features to float array
        features = np.array([[float(data.get(k) or 0) for k in feature_keys]])
        prediction = model.predict(features)[0]

        result = "Malignant" if prediction == 1 else "Benign"

        # Prepare record for MongoDB
        record = {
            "personal": {
                "name": data.get("name", "Unknown"),
                "age": data.get("age", "N/A"),
                "gender": data.get("gender", "N/A"),
            },
            "features": {k: data.get(k, None) for k in feature_keys},
            "prediction": result,
            "timestamp": datetime.utcnow()
        }
        print("Record being inserted:", record) 

        # Insert record into MongoDB
        patients_collection.insert_one(record)
        print("✅ Record inserted successfully into 'diagnosis' collection.")
        return jsonify({"prediction": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 2️⃣ Route to fetch all stored patients

@app.route("/records", methods=["GET"])
def get_all_patients():
    try:
        # If DB not configured, return informative 503
        if patients_collection is None:
            return jsonify({"error": "MongoDB is not configured. Set a valid MONGO_URI in .env and ensure Atlas network access."}), 503

        # Fetch all patients with basic info only
        records = list(patients_collection.find({}, {"personal.name": 1, "personal.age": 1, "personal.gender": 1, "prediction": 1}))
        for r in records:
            r["_id"]=str(r["_id"])
        return jsonify({"Patients": records}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 2️⃣ Route to fetch details of one patient by name
@app.route("/records/<id>", methods=["GET"])
def get_patient_details(id):
    try:
        if patients_collection is None:
            return jsonify({"error": "MongoDB is not configured. Set a valid MONGO_URI in .env and ensure Atlas network access."}), 503

        record = patients_collection.find_one({"_id": ObjectId(id)})
        if record:
            record["_id"]= str(record["_id"])
            return jsonify({"Patient": record}), 200
        else:
            return jsonify({"message": f"No record found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
