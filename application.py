# application.py
from flask import Flask, request, render_template, jsonify
import numpy as np
import os
import logging
import joblib

# Prefer tensorflow.keras to avoid importing standalone keras.wrappers
try:
    from tensorflow.keras.models import load_model
    backend = "tensorflow.keras"
except Exception:
    from keras.models import load_model
    backend = "keras"

# Setup base & templates
basedir = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
templates_dir = os.path.join(basedir, "templates")
os.makedirs(templates_dir, exist_ok=True)

application = Flask(__name__, template_folder=templates_dir)
app = application

# Logging
logging.basicConfig(level=logging.INFO)
app.logger.info(f"Using Keras backend: {backend}")

# Paths to saved artifacts
models_dir = os.path.join(basedir, 'models')
scaler_path = os.path.join(models_dir, 'scaler.save')
model_path = os.path.join(models_dir, 'BankExit_predict.h5')

# Load scaler (joblib)
scaler = None
try:
    scaler = joblib.load(scaler_path)
    app.logger.info(f"Loaded scaler from: {scaler_path}")
except Exception as e:
    app.logger.exception(f"Failed to load scaler from {scaler_path}: {e}")
    scaler = None

# Load Keras model
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

# Diagnostic routes
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/routes", methods=["GET"])
def routes():
    # helpful for debugging route map
    return jsonify([rule.rule for rule in app.url_map.iter_rules()])

# JSON API route (easy to test)
@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    Expect JSON body: keys -> credit, geo, gen, age, ten, bal, prod, card, act, salary
    Returns JSON: {"score": float, "prediction": "exit"|"stay"}
    """
    if model is None:
        return {"error": "Model not loaded on server. Check logs."}, 500

    data = request.get_json(silent=True)
    if not data:
        return {"error": "Please POST JSON body"}, 400

    try:
        keys = ['credit','geo','gen','age','ten','bal','prod','card','act','salary']
        vals = []
        for k in keys:
            v = data.get(k, 0)
            vals.append(float(v))

        input_features = np.array([vals], dtype=np.float32)

        if scaler is not None:
            X = scaler.transform(input_features)
        else:
            X = input_features

        pred = model.predict(X)
        score = float(np.array(pred).ravel()[0])
        msg = "exit" if score > 0.5 else "stay"
        return {"score": score, "prediction": msg}, 200

    except Exception as e:
        app.logger.exception("Error in /predict_api")
        return {"error": str(e)}, 500

# HTML form route
@app.route("/", methods=['GET', 'POST'])
def predict_bank_exit():
    prediction_text = ""
    if request.method == "POST":
        # If model not loaded, friendly message
        if model is None:
            prediction_text = "Model failed to load on server. Check server logs."
            return render_template('index.html', prediction_result=prediction_text)

        try:
            def safe_float(name, default=0.0):
                v = request.form.get(name, "")
                if v is None or v == "":
                    return default
                try:
                    return float(v)
                except ValueError:
                    app.logger.warning(f"Invalid float for {name}: {v}")
                    return default

            # Parse inputs (order must match training)
            credit = safe_float('credit')
            geo    = safe_float('geo')
            gen    = safe_float('gen')
            age    = safe_float('age')
            ten    = safe_float('ten')
            bal    = safe_float('bal')
            prod   = safe_float('prod')
            card   = safe_float('card')
            act    = safe_float('act')
            salary = safe_float('salary')

            input_features = np.array([[credit, geo, gen, age, ten, bal, prod, card, act, salary]], dtype=np.float32)
            app.logger.debug(f"Raw input: {input_features}")

            if scaler is not None:
                X = scaler.transform(input_features)
            else:
                app.logger.warning("No scaler loaded — using raw inputs as fallback")
                X = input_features

            app.logger.debug(f"Scaled input: {X}, shape: {X.shape}")

            result = model.predict(X)
            app.logger.debug(f"Model output: {result}")

            score = float(np.array(result).ravel()[0])
            prediction_text = "The customer is likely to exit." if score > 0.5 else "The customer is likely to stay."

        except Exception as e:
            app.logger.exception("Error during prediction")
            prediction_text = f"An error occurred during prediction: {e}"

    return render_template('index.html', prediction_result=prediction_text)

if __name__ == "__main__":
    # Start server
    app.run(host="0.0.0.0", port=5000, debug=True)
