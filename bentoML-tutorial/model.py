import bentoml
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()
X = np.array([[0,0], [1,1]])
y = [0, 1]
model.fit(X, y)

bentoml.sklearn.save_model("my_model", model)