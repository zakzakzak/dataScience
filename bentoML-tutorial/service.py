import bentoml
import numpy as np

@bentoml.service
class MyService:

    def __init__(self):
        # load model sekali saja (init)
        self.model = bentoml.sklearn.load_model("my_model:latest")

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.predict(input_data)