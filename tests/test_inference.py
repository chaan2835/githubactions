import pickle
import numpy as np

def test_model_prediction():
    with open("model/xgbregressor_model.pkl", "rb") as f:
        model = pickle.load(f)

    dummy_input = np.array([[100, 20, 30]])  # Example input
    prediction = model.predict(dummy_input)

    assert prediction is not None
    assert isinstance(prediction[0], float) or isinstance(prediction[0], np.floating)
