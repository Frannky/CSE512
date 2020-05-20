import argparse
import csv
import numpy as np
from math import sqrt
from sys import exit

def generate_data(dataset, split):
    data = []
    with open(dataset, 'r') as data_file:
        data_file.readline()
        reader = csv.reader(data_file)
        for row in reader:
            x = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
            if int(row[5]) == 1:
                y = 1
            else:
                y = -1
            data.append((x, y))
    cut = int(len(data) * split)
    return data[:cut], data[cut:]

def euclidean_distance(r1,r2):
    dist = 0.0
    for i in range(len(r1)):
        dist += (r1[i] - r2[i]) ** 2
    return sqrt(dist)

def test(training_set, testing_set, k):
    count = 0
    for test in testing_set:
        result = []
        temp_list = []
        for (x, y) in training_set:
            dist = euclidean_distance(test[0],x)
            temp_list.append((dist, y))
        temp_list.sort(key=lambda item: item[0])
        for i in range(k):
            result.append(temp_list[i][1])
        predict = max(set(result), key=result.count)
        if predict > 0:
            predict = 1
        else:
            predict = -1
        if predict != test[1]:
            count += 1
    return count / len(testing_set)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="/path/to/data/filename.csv", required=True)
    parser.add_argument("-k", "--k-val", help="k value of knn", required=True, type=int)
    parser.add_argument("-s", "--split", help="this number is the percentage of training data of all the dataset, default = 0.8", default=0.8, type=float)
    args = parser.parse_args()
    dataset = args.dataset
    split = args.split
    k = args.k_val
    if split > 1:
        print("Split value cannot exceed 1.")
        exit(1)
    if k <= 0:
        print("K value cannot be less than 1.")
        exit(1)
    return dataset, split, k

def main():
    dataset, split, k = parse_args()
    training_set, testing_set = generate_data(dataset, split)
    error = test(training_set, testing_set, k)
    print('Dataset path is: {}'.format(dataset))
    print('Training set has {}% of total dataset'.format(split * 100))
    print('This is {}-nearest-neighbors!'.format(k))
    print("the error is", error)

if __name__ == '__main__':
    main()
