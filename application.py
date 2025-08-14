from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model

# We are not using the 'os' or 'pickle'/'joblib' modules.

application = Flask(__name__)
app = application

# --- File Loading with Relative Path ---
# This path assumes you run the script from your project's root directory,
# and the model is inside a 'models' subfolder.
model_path = 'models/BankExit_predict.h5'

# Load the Keras model
model = load_model(model_path)

# --- Manual Scaling Values ---
# IMPORTANT: These values are critical for correct predictions.
# You MUST replace these placeholders with the actual mean and scale values 
# printed from your training notebook.
MEANS = np.array([ 2.48689958e-17,  4.08562073e-17,  1.42108547e-17, -8.88178420e-19,
  2.39808173e-17, -2.79776202e-17, -8.88178420e-18, -1.42108547e-17,
 -2.93098879e-17,  1.42108547e-17])
SCALES = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])


# --- Main application route for handling the web form ---
@app.route("/", methods=['GET', 'POST'])
def predict_bank_exit():
    prediction_text = ""
    if request.method == "POST":
        try:
            # Safely parse all form inputs, converting them to floats.
            # This prevents crashes if fields are empty or contain non-numeric text.
            credit = float(request.form.get('credit', 0))
            geo = float(request.form.get('geo', 0))
            gen = float(request.form.get('gen', 0))
            age = float(request.form.get('age', 0))
            ten = float(request.form.get('ten', 0))
            bal = float(request.form.get('bal', 0))
            prod = float(request.form.get('prod', 0))
            card = float(request.form.get('card', 0))
            act = float(request.form.get('act', 0))
            salary = float(request.form.get('salary', 0))

            # Assemble the features into a 2D numpy array for the model.
            input_features = np.array([[credit, geo, gen, age, ten, bal, prod, card, act, salary]])

            # Manually scale the input features using the pre-calculated statistics.
            new_data_scaled = (input_features - MEANS) / SCALES

            # Use the loaded Keras model to generate a prediction probability.
            result = model.predict(new_data_scaled)

            # Interpret the model's raw output probability (a value from 0 to 1).
            if result[0][0] > 0.5:
                prediction_text = "Prediction: The customer is likely to exit."
            else:
                prediction_text = "Prediction: The customer is likely to stay."

        except Exception as e:
            # Provide a user-friendly error message if anything goes wrong.
            prediction_text = f"An error occurred during prediction: {e}"
    
    # Render the main page, passing the prediction result back to the user.
    return render_template('index.html', prediction_result=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
