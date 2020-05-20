import argparse
import csv
import re

import numpy as np
from random import seed
from random import randrange
import math
import sys

classifiers = []
num_of_classifier = 8

class Stump_Node():
    """
    """
    direction = 1
    index = None
    border = None
    weight = None

def predict(x):
    """
    """
    num_of_data = len(x)
    prediction = np.array([0 for i in range(num_of_data)])
    for classifier in classifiers:
        pred = np.array([1 for i in range(num_of_data)])
        init = classifier.direction * x[:,classifier.index]
        thresh = classifier.direction * classifier.border
        index_below = init < thresh
        pred[index_below] = -1
        prediction = prediction + classifier.weight * pred
    prediction = np.sign(prediction).flatten()
    return prediction

def adaboosting(x, y):
    """
    """
    num_of_data = len(y)
    num_of_feature = len(np.transpose(x))
    w = np.array([1 / num_of_data for i in range(num_of_data)])
    global classifiers
    global num_of_classifier
    weights = []

    for i in range(num_of_classifier):
        classifier = Stump_Node()
        epsilon = sys.maxsize
        for feature in range(num_of_feature):
            feature_column = np.reshape(x[:, feature], (-1,1))
            unique_feature = np.unique(feature_column)
            sorted(unique_feature)
            for value in unique_feature:
                direction = 1
                predictions = np.array([1 for i in range(num_of_data)])
                predictions[x[:, feature] < value] = -1
                misclassified = sum(w[y != predictions])
                if misclassified > 0.5:
                    misclassified = 1 - misclassified
                    direction = -1
                if misclassified < epsilon:
                    epsilon = misclassified
                    classifier.direction = direction
                    classifier.index = feature
                    classifier.border = value
        classifier.weight = math.log((1.0) / (epsilon + 1e-11) - 1.0) * 0.5
        pred = np.array([1 for i in range(len(y))])
        init = classifier.direction * x[:, classifier.index]
        thresh = classifier.direction * classifier.border
        index_below = init < thresh
        pred[index_below] = -1
        w = w * np.exp(-classifier.weight * y * pred)
        w = w / np.sum(w)
        classifiers.append(classifier)
        weights.append(classifier.weight)
    return weights, classifiers


def load_csv(file_name):
    """
    """
    print(file_name)
    with open(file_name, 'r') as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        row_count = 0
        data_arrays = [[0 for i in range(569)] for j in range(6)]
        for row in csv_reader:
            if row_count >= 1:
                column_count = 0
                for data in row:
                    data_arrays[column_count][row_count - 1] = float(data.strip())
                    column_count += 1
            row_count += 1
    return data_arrays


def regularize(y):
    """
    """
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1.0


def cross_validation(data, k_folds):
    """
    """
    data = np.transpose(data)
    split_data = list()
    data_copy = list(data)
    data_number_each_set = int(len(data) / k_folds)
    for i in range(k_folds):
        current = []
        while len(current) < data_number_each_set:
            index = randrange(len(data_copy))
            current.append(data_copy.pop(index).tolist())
        split_data.append(current)
    return split_data


def get_score(observed_y, pred_y):
    """
    """
    count = 0
    for i in range(len(observed_y)):
        if observed_y[i] * pred_y[i] > 0:
            count += 1
    if len(observed_y) == 0: return
    return 1 - count / len(observed_y)


def algorithm_evaluate(data, k_folds):
    """
    """
    global classifiers
    n = len(data)-1
    scores = list()
    weights = list()
    split_data = cross_validation(data, k_folds)
    for fold in split_data:
        train_set = list(split_data)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        train_set = np.transpose(train_set)
        test_set = np.transpose(test_set)
        train_x = train_set[0:n]
        train_y = train_set[n]
        test_x = test_set[0:n]
        weight, classifiers = adaboosting(np.transpose(train_x), train_y)
        predictions = predict(np.transpose(test_x))
        observed = [row[-1] for row in fold]
        score = get_score(observed, predictions)
        scores.append(score)
        weights.append(weight)
    return scores,weights

def argument():
    """

    :return:
    """
    def mode(arg):
        match = re.compile('erm|[1-9]\d*-fold').match(arg)
        if match:
            return arg
        else:
            raise argparse.ArgumentTypeError

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="/path/to/data/filename.csv", required=True)
    parser.add_argument("-m", "--mode", help="either erm or <int>-fold", type=mode, required=True)
    parser.add_argument("-noc", "--number_of_classifier", default=8, help="number of classifier(must be integer)", type=int)

    return parser.parse_args()

def main():
    """

    :return:
    """
    seed(12)
    global classifiers
    global num_of_classifier
    arguments = argument()
    data_arrays = load_csv(arguments.dataset)
    x1 = data_arrays[0]
    x2 = data_arrays[1]
    x3 = data_arrays[2]
    x4 = data_arrays[3]
    x5 = data_arrays[4]
    y = data_arrays[5]
    regularize(y)
    data = np.array([x1, x2, x3, x4, x5, y])
    k_folds = 10
    scores = list()
    weights = list()

    if arguments.number_of_classifier:
        num_of_classifier = arguments.number_of_classifier
    else:
        num_of_classifier = 8

    if arguments.mode == 'erm':
        n = len(data) -1
        weight, classifiers = adaboosting(np.transpose(data[0:n]), np.transpose(data[n]))
        predictions = predict(np.transpose(data[0:n]))
        observed = np.transpose(data[n])
        score = get_score(observed, predictions)
        scores.append(score)
        weights.append(weight)
    else:
        scores, weights = algorithm_evaluate(data, k_folds)
    print('Weights: %s' % weights)
    print('Errors: %s' % scores)
    # print('number of classifier: ', num_of_classifier)
    # print('Mean Error value: %.10f' % (sum(scores) / float(len(scores))))
    print('Mean Error: %.3f%%' % (sum(scores) * 100 / float(len(scores))))
    # print('Argument List:', str(sys.argv))

if __name__ == '__main__':
    main()