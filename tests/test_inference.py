# tests/test_inference.py

import os
import pickle
import numpy as np

def test_model_prediction_output():
    model_path = "model/xgbregressor_model.pkl"

    # Ensure model file exists
    assert os.path.exists(model_path), "Trained model not found!"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Test inference with dummy input (based on your feature set)
    dummy_input = np.array([[100, 20, 30, 150, 2000, 3000, 600, 10000, 400, 900]])
    prediction = model.predict(dummy_input)

    # Check output shape and value type
    assert prediction is not None, "Model prediction is None"
    assert len(prediction) == 1, "Prediction output shape is incorrect"
    assert isinstance(prediction[0], float), "Prediction output is not a float"
    assert prediction[0] > 0, "Prediction should be greater than zero"
