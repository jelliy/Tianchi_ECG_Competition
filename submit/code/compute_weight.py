import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pickle
import torch, time, os, shutil
import models, utils
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_result(y, p):
    r = confusion_matrix(y, p, labels=[1.0, 0.0])
    print(r)
    # print(accuracy_score(y, p))
    print(f1_score(y, p))
    return r

def calc_f1_all(y_true, y_pre):
    cm = np.zeros((2, 2))
    for i in range(34):
        print("------- " + str(i) + " --------")
        predict = np.array([0. if i < 0.5 else 1. for i in y_pre[:, i]])
        r = get_result(y_true[:, i], predict)
        cm += r
    print(cm)
    p = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    q = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print(2 * p * q / (p + q))

r1 = pd.read_csv("r1.csv").values
print(r1)
r2 = pd.read_csv("r2.csv").values
print(r2)
y_all = pd.read_csv("y.csv").values
#print(y)
r = r1*0.52+r2*0.48

#calc_f1_all(y_all,r)

def compute_model_weight(r1,r2):
    X = r1 - r2
    Y = y_all - r2
    w1 = []
    for i in range(34):
        x = X[:, i]
        y = Y[:, i]
        w1.append(np.dot(x,y)/np.dot(x,x))
    w1 = np.array(w1)
    w2 = 1-w1
    calc_f1_all(y_all,w1*r1+w2*r2)
    print(w1)
    return w1

compute_model_weight(r1, r2)