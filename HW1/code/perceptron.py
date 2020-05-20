import csv
import numpy as np
from random import seed
from random import randrange
import sys
import re
import argparse


def load_csv(file_name):
    """
    """
    if 'linearly-separable-dataset.csv' in file_name:
        with open(file_name, 'r') as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            row_count = 0
            data_arrays = [[0 for i in range(1151)] for j in range(3)]
            for row in csv_reader:
                if row_count >= 1:
                    column_count = 0
                    for data in row:
                        data_arrays[column_count][row_count - 1] = float(data.strip())
                        column_count += 1
                row_count += 1
        return data_arrays
    elif 'Breast_cancer_data.csv' in file_name:
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
    else:
        print('unrecognized data')
        return

def regularize(y):
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1.0

def calculate_weight(data, learning_rate, iterations):
    """
    """
    w = [0 for i in range(6)]
    n = 0
    temp = np.transpose(data)
    x = np.transpose(temp[0:6])
    y = np.transpose(temp[6])
    flag = 1

    while True:
        for i in range(len(x)):
            f = np.dot(w , x[i]) * y[i]
            if f <= 0:
                flag = 0
                for j in range(len(w)):
                    w[j] = w[j] + learning_rate * y[i] * x[i][j]
                    n += 1
        if flag == 1 or n > iterations:
            break
        else:
            flag = 1
    return w

def perceptron(train, test, learning_rate, num):
    """
    """
    predictions = []
    test = np.transpose(test)
    test = test[:6]
    test = np.transpose(test)
    w = calculate_weight(train, learning_rate,num)
    for row in test:
        f = np.dot(w,row)
        predictions.append(f)
    return predictions,w

def perceptron_erm(train, test, learning_rate, num):
    """
    """
    predictions = []
    test = test[:6]
    test = np.transpose(test)
    w = calculate_weight(np.transpose(train), learning_rate,num)
    for row in test:
        f = np.dot(w,row)
        predictions.append(f)
    return predictions,w

def calculate_weight_linear(data, learning_rate, iterations):
    """
    """
    w = [0 for i in range(3)]
    n = 0
    temp = np.transpose(data)
    x = np.transpose(temp[0:3])
    y = np.transpose(temp[3])
    flag = 1
    while True:
        for i in range(len(x)):
            f = np.dot(w, x[i]) * y[i]
            if f <= 0:
                flag = 0
                for j in range(len(w)):
                    w[j] = w[j] + learning_rate * y[i] * x[i][j]
                    n += 1

        if flag == 1 or n > iterations:
            break
        else:
            flag = 1
    return w


def linear_perceptron(train, test, learning_rate, num):
    """
    """
    predictions = []
    test = np.transpose(test)
    test = test[:3]
    test = np.transpose(test)
    w = calculate_weight_linear(train, learning_rate, num)
    for row in test:
        f = np.dot(w, row)
        predictions.append(f)
    return predictions, w

def linear_perceptron_erm(train, test, learning_rate, num):
    """
    """
    predictions = []
    test = test[:3]
    test = np.transpose(test)
    w = calculate_weight_linear(np.transpose(train), learning_rate, num)
    for row in test:
        f = np.dot(w, row)
        predictions.append(f)
    return predictions, w


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


def algorithm_evaluate(data, algorithm, k_folds, learning_rate, num):
    """
    """
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
        predictions, w = algorithm(train_set, test_set, learning_rate, num)
        observed = [row[-1] for row in fold]
        score = get_score(observed, predictions)
        scores.append(score)
        weights.append(w)
    return scores, weights


def argument():
    def mode(arg):
        match = re.compile('erm|[1-9]\d*-fold').match(arg)
        if match:
            return arg
        else:
            raise argparse.ArgumentTypeError

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="/path/to/data/filename.csv", required=True)
    parser.add_argument("-m", "--mode", help="either erm or <int>-fold", type=mode, required=True)
    parser.add_argument("-mi", "--max_iterations", default=1000000, help="max number of iterations for linear non-separate dataset(must be integer)", type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.1, help="learning rate of perceptron algorithm(float number)", type=float)

    return parser.parse_args()


def main():

    arguments = argument()
    data_arrays = load_csv(arguments.dataset)
    if 'linearly-separable-dataset.csv' in arguments.dataset:
        seed(1)
        x0 = [1 for i in range(1151)]
        x1 = data_arrays[0]
        x2 = data_arrays[1]
        y = data_arrays[2]
        regularize(y)
        data = [x0, x1, x2, y]
    elif 'Breast_cancer_data.csv' in arguments.dataset:
        seed(13)
        x0 = [1 for i in range(569)]
        x1 = data_arrays[0]
        x2 = data_arrays[1]
        x3 = data_arrays[2]
        x4 = data_arrays[3]
        x5 = data_arrays[4]
        y = data_arrays[5]
        regularize(y)
        data = [x0, x1, x2, x3, x4, x5, y]
    else:
        print("unrecognized data")
        return
    k_folds = 10
    num = 10000000
    learning_rate = 0.1
    scores = list()
    weights = list()
    if arguments.learning_rate:
        learning_rate = arguments.learning_rate
    else:
        learning_rate = 0.1

    if arguments.max_iterations:
        num = arguments.max_iterations
    else:
        num = 10000

    if 'linearly-separable-dataset.csv' in arguments.dataset:
        if arguments.mode == 'erm':
            predictions, w = linear_perceptron_erm(data, data, learning_rate, num)
            observed = y
            score = get_score(observed, predictions)
            scores.append(score)
            weights.append(w)
        else:
            scores, weights = algorithm_evaluate(data, linear_perceptron, k_folds, learning_rate, num)
    elif 'Breast_cancer_data.csv' in arguments.dataset:
        if arguments.mode == 'erm':
            predictions, w = perceptron_erm(data, data, learning_rate, num)
            observed = y
            score = get_score(observed, predictions)
            scores.append(score)
            weights.append(w)
        else:
            scores, weights = algorithm_evaluate(data, perceptron, k_folds, learning_rate, num)

    print('Weights: %s' % weights)
    print('Errors: %s' % scores)
    print('Mean Error: %.3f%%' % (sum(scores) * 100/ float(len(scores))))

if __name__ == '__main__':
    main()
