# test_load.py
import os, joblib, numpy as np
from tensorflow.keras.models import load_model

basedir = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(basedir, 'models')
print("models dir:", models_dir)
print("files:", os.listdir(models_dir))

scaler = joblib.load(os.path.join(models_dir, 'scaler.save'))
print("scaler mean:", scaler.mean_, "scale:", scaler.scale_)

m = load_model(os.path.join(models_dir, 'BankExit_predict.h5'))
print("model input shape:", m.input_shape)
# test predict with a dummy sample (shape must match)
dummy = np.zeros((1, m.input_shape[1]), dtype=np.float32)
print("predict:", m.predict(dummy))
