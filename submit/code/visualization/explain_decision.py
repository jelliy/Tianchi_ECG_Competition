"""
Reads answers.txt file and REFERENCE.csv file
and compares correct labes with predicted
Than outputs the list of wrongly classified training samples

NOTE:
    this script provides you an ability to plot wrongly classified entries
NOTE:
    make sure you have generated the answers.txt file
"""

import matplotlib
import pydotplus
import sklearn
from sklearn.externals import joblib

import preprocessing
from features import feature_extractor5
from loading import loader
from preprocessing import normalizer

matplotlib.use("Qt5Agg")

import numpy as np

model_file = "../model.pkl"

name = input("Enter an entry name to plot: ")
name = name.strip()
if len(name) == 0:
    print("Finishing")
    exit(-1)

if not loader.check_has_example(name):
    print("File Not Found")
    exit(-1)

x = loader.load_data_from_file(name, "../validation")
x = normalizer.normalize_ecg(x)
feature_names = feature_extractor5.get_feature_names(x)
x = feature_extractor5.features_for_row(x)

# as we have one sample at a time to predict, we should resample it into 2d array to classify
x = np.array(x).reshape(1, -1)

model = joblib.load(model_file)
dtree = model.estimators_[0]


def export_tree(dtree, feature_names):
    dot_data = sklearn.tree.export_graphviz(dtree, feature_names=feature_names, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("dtree.png")


def explain(tree, x):
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    node_indicator = tree.decision_path(x)

    for node_id in node_indicator.indices:
        if (x[0, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
              % (node_id,
                 0,
                 feature_names[feature[node_id]],
                 x[0, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

    print(tree.predict(x))


def get_code(tree, feature_names=None, tabdepth=0):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    if feature_names is None:
        features = ['f%d' % i for i in tree.tree_.feature]
    else:
        features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, tabdepth=0):
        if (threshold[node] != -2):
            print('\t' * tabdepth,
                  "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], tabdepth + 1)
            print('\t' * tabdepth,
                  "} else {")
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node], tabdepth + 1)
            print('\t' * tabdepth,
                  "}")
        else:
            print('\t' * tabdepth,
                  "return " + str(value[node]))

    recurse(left, right, threshold, features, 0)

explain(dtree, x)
