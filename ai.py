from tabnanny import verbose
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

a = pd.read_csv("train2.csv")
scaler = MinMaxScaler()


y = a["target"]
#print(a)
x = a.drop("target", axis=1)
scaler.fit(x)
x = scaler.transform(x)

model = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(62, 36, 18, 9, 3), random_state=1, verbose=1, max_iter=1000)
model.fit(x, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)