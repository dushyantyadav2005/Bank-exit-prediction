# application.py
from flask import Flask, request, render_template, jsonify
import numpy as np
import logging
import joblib

# Prefer tensorflow.keras to avoid importing standalone keras
try:
    from tensorflow.keras.models import load_model
    backend = "tensorflow.keras"
except Exception:
    from keras.models import load_model
    backend = "keras"

# Initialize Flask (template folder is directly specified)
application = Flask(__name__, template_folder="templates")
app = application

# Logging
logging.basicConfig(level=logging.INFO)
app.logger.info(f"Using Keras backend: {backend}")

# Paths to saved artifacts (relative paths)
scaler_path = "models/scaler.save"
model_path = "models/BankExit_predict.h5"

# Load scaler
scaler = None
try:
    scaler = joblib.load(scaler_path)
    app.logger.info(f"Loaded scaler from: {scaler_path}")
except Exception as e:
    app.logger.exception(f"Failed to load scaler from {scaler_path}: {e}")
    scaler = None

# Load model
model = None
try:
    model = load_model(model_path)
    app.logger.info(f"Loaded model from: {model_path}")
    try:
        app.logger.info(f"Model input shape: {model.input_shape}")
    except Exception:
        pass
except Exception as e:
    app.logger.exception(f"Failed to load model from {model_path}: {e}")
    model = None

# Health check
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

# Route list for debugging
@app.route("/routes", methods=["GET"])
def routes():
    return jsonify([rule.rule for rule in app.url_map.iter_rules()])

# API endpoint for JSON requests
@app.route("/predict_api", methods=["POST"])
def predict_api():
    if model is None:
        return {"error": "Model not loaded on server. Check logs."}, 500

    data = request.get_json(silent=True)
    if not data:
        return {"error": "Please POST JSON body"}, 400

    try:
        keys = ['credit', 'geo', 'gen', 'age', 'ten', 'bal', 'prod', 'card', 'act', 'salary']
        vals = [float(data.get(k, 0)) for k in keys]

        input_features = np.array([vals], dtype=np.float32)
        X = scaler.transform(input_features) if scaler else input_features

        pred = model.predict(X)
        score = float(pred.ravel()[0])
        msg = "exit" if score > 0.5 else "stay"
        return {"score": score, "prediction": msg}, 200

    except Exception as e:
        app.logger.exception("Error in /predict_api")
        return {"error": str(e)}, 500

# HTML form endpoint
@app.route("/", methods=["GET", "POST"])
def predict_bank_exit():
    prediction_text = ""
    if request.method == "POST":
        if model is None:
            prediction_text = "Model failed to load on server. Check server logs."
            return render_template('index.html', prediction_result=prediction_text)

        try:
            def safe_float(name, default=0.0):
                try:
                    return float(request.form.get(name, default))
                except ValueError:
                    app.logger.warning(f"Invalid float for {name}")
                    return default

            # Gather input
            credit = safe_float('credit')
            geo = safe_float('geo')
            gen = safe_float('gen')
            age = safe_float('age')
            ten = safe_float('ten')
            bal = safe_float('bal')
            prod = safe_float('prod')
            card = safe_float('card')
            act = safe_float('act')
            salary = safe_float('salary')

            input_features = np.array([[credit, geo, gen, age, ten, bal, prod, card, act, salary]], dtype=np.float32)
            X = scaler.transform(input_features) if scaler else input_features

            result = model.predict(X)
            score = float(result.ravel()[0])
            prediction_text = "The customer is likely to exit." if score > 0.5 else "The customer is likely to stay."

        except Exception as e:
            app.logger.exception("Error during prediction")
            prediction_text = f"An error occurred during prediction: {e}"

    return render_template('index.html', prediction_result=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
