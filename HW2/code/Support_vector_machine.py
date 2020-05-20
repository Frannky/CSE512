import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import argparse

def generate_data():
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    X1 = np.c_[np.ones((X0.shape[0])), X0]
    positive_x =[]
    negative_x =[]
    for i,label in enumerate(y):
        if label == 0:
            negative_x.append(X1[i])
        else:
            positive_x.append(X1[i])
    data_dict = {-1:np.array(negative_x), 1:np.array(positive_x)}
    max_fval = float("-inf")
    min_fval = float("inf")
    for y_i in data_dict:
        if np.amax(data_dict[y_i]) > max_fval:
            max_fval=np.amax(data_dict[y_i])
        if np.amin(data_dict[y_i]) < min_fval:
            min_fval=np.amin(data_dict[y_i])
    step_sizes = [max_fval * 0.1, max_fval * 0.01, max_fval * 0.001,...]
    return X1, y, data_dict, step_sizes, max_fval, min_fval

def split_data(data):
    train_data = {-1:data[-1][0:int(len(data[-1]) * 0.8)], 1: data[1][0:int(len(data[1]) * 0.8)]}
    test_data = {-1:data[-1][int(len(data[-1]) * 0.8):], 1:data[1][int(len(data[1]) * 0.8):]}
    return train_data, test_data

def train(data_dict):
    global T
    global lambda1
    size = len(data_dict[1][0])
    max_fval = float('-inf')
    for y_i in data_dict:
        if np.amax(data_dict[y_i]) > max_fval:
            max_fval = np.amax(data_dict[y_i])
    b = 0
    theta = [0.0 for i in range(size)]
    w = [[0.0] * size for _ in range(T+1)]
    b_value = [0.0 for _ in range(T+1)]
    step_sizes = [max_fval * 0.1, max_fval * 0.01, max_fval * 0.001]
    for step_size in step_sizes:
        for t in range(1,T):
            for index in range(80):
                w[t] = [i for i in theta]
                b_value[t] = b
                if 0 <= index < 40:
                    i = index
                    y_i = -1
                    x_i = data_dict[y_i][i]
                    if y_i * (np.dot(w[t], x_i) + b) < 1:
                        theta = theta + step_size / t * (y_i * x_i / lambda1 - theta)
                        theta[0] = 0
                        b += step_size / t * (y_i / lambda1 - b)
                    else:
                        theta = theta * (1 - step_size / t)
                        theta[0] = 0
                else:
                    i = index - 40
                    y_i = 1
                    x_i = data_dict[y_i][i]
                    if y_i * (np.dot(w[t], x_i) + b) < 1:
                        theta = theta + step_size / t * (y_i * x_i / lambda1 - theta)
                        theta[0] = 0
                        b += step_size / t * (y_i / lambda1 - b)
                    else:
                        theta = theta * (1 - step_size / t)
                        theta[0] = 0
    result = np.average(w, axis=0)
    result[0] = np.average(b_value, axis=0)
    return result

def test(data_dict, w):
    if data_dict == None:
        return
    count = 0
    sum = 0
    for y_i in data_dict:
        for x_i in data_dict[y_i]:
            if y_i * np.dot(np.array(x_i), w) > 0:
                count += 1
            sum += 1
    return 1 - count/sum

def draw(data_dict, w, max_fval, min_fval):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim([-1, 9])
    plt.ylim([-13, 9])
    w1 = w[1:]
    b1 = w[0]
    X1 = []
    y = []
    for y_i in data_dict:
        for x_i in data_dict[y_i]:
            X1.append(x_i)
            y.append(y_i)
    X1 = np.array(X1)
    plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=y)
    min = min_fval - 2
    max = max_fval + 2
    positive_min = hyperplane(min, w1, b1,  1)
    positive_max = hyperplane(max, w1, b1,  1)
    ax.plot([min, max], [positive_min, positive_max], 'd--')
    negative_min = hyperplane(min, w1, b1,  -1)
    negative_max = hyperplane(max, w1, b1,  -1)
    ax.plot([min, max], [negative_min, negative_max], 'd--')
    svm_min = hyperplane(min, w1, b1,  0)
    svm_max = hyperplane(max, w1, b1,  0)
    ax.plot([min, max], [svm_min, svm_max], 'black')
    if lambda1 == 0.02:
        plt.scatter(3.66, 1.9, marker='o', facecolors='none', edgecolors='black', s = 150)
        plt.scatter(5.72, -7.05, marker='o', facecolors='none', edgecolors='black', s = 150)
    plt.title("SVM")
    plt.show()

def hyperplane(x, w, b, margin):
    return (-w[0] * x - b + margin) / w[1]

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mi", "--max_iterations", default=10000,
                        help="max number of iterations for linear non-separate dataset(must be integer)", type=int)
    parser.add_argument("-l", "--lambda1", default=0.02,
                        help="lambda value of svm algorithm(float number)", type=float)
    return parser.parse_args()

def main():
    arguments = argument()
    global lambda1
    global T
    if arguments.lambda1:
        lambda1 = arguments.lambda1
    else:
        lambda1 = 0.02
    if arguments.max_iterations:
        T = arguments.max_iterations
    else:
        T = 10000
    X1, y, data, step_size, max_fval, min_fval = generate_data()
    train_data, test_data = split_data(data)
    w = train(train_data)
    print("the maximum iteration you choose is:", T)
    print("the value of lambda you choose is:", lambda1)
    print('Weights: %s' % w)
    draw(train_data, w, max_fval, min_fval)
    error = test(test_data, w)
    print('Errors: %s' % error)

if __name__ == '__main__':
    main()
