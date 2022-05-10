import pandas as pd
import numpy as np
#from sklearn.model_selection import MLPClassifier


dict = {"a":1, "b":2, "c":3, "d":4, "e":5, "f":6, "g":7, "h":8, "i":9, "j":10, "k":11, "l":12, "m":13, "n":14, "o":15, "p":16, "q":17, "r":18, "s":19, "t":20, "u":21, "v":22, "w":23, "x":24, "y":25, "z":26}
a = pd.read_csv("train.csv")


n = -1
for i in a["f_27"]:
    b = []
    n += 1
    i = list(i)
    for j in i:
        j = j.lower()
        b.append(dict[j])
    b = np.array(b)
    b = sum(b)
    a.loc[n, "f_27"] = b
    del b


print(a)
a.to_csv("train2.csv")

