import argparse
import csv
import numpy as np
from math import sqrt
from sys import exit
from random import seed
from random import randrange


def generate_data(dataset):
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
    return data


def converge(pre_center, cur_center, threshold):
    if np.sum((pre_center - cur_center) / pre_center * 100) > threshold:
        return False
    return True


def euclidean_distance(r1, r2):
    dist = 0.0
    for i in range(len(r1)):
        dist += (r1[i] - r2[i]) ** 2
    return sqrt(dist)


def manhatten_distance(r1, r2):
    return np.abs(r1 - r2)


def k_means(dataset, k, distance, max_iterations, threshold):
    result = [{} for _ in range(k)]
    for group in result:
        index = randrange(len(dataset))
        group['center'] = dataset[index][0]
        group['data_points'] = []
    t = 0
    while t < max_iterations:
        for group in result:
            group['data_points'] = []
        for x, y in dataset:
            if distance == 'Manhattan':
                group_index = int(np.argmin([manhatten_distance(x, group['center']) for group in result]))
            else:
                group_index = int(np.argmin([euclidean_distance(x, group['center']) for group in result]))
            result[group_index]['data_points'].append((x, y))
        isConverge = 1
        for group in result:
            cur_center = np.average([x[0] for x in group['data_points']], axis=0)
            pre_center = group['center']
            if not converge(pre_center, cur_center, threshold):
                isConverge = 0
            group['center'] = cur_center
        if isConverge == 1:
            return result, 1
        t += 1
    return result, 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="/path/to/data/filename.csv", required=True)
    parser.add_argument("-k", "--k-val", help="This is th number of clusters value chosen by the user, default=2", default=2, type=int)
    parser.add_argument("--distance", help="This is the distance choice provided to user to choose between Euclidean "
                                           "distance and Manhattan distance. The default value is Euclidean distance.", default="Euclidean")
    parser.add_argument("--max-iterations", help="This is the maximum iterations of k-means and default=500.", default=500, type=int)
    parser.add_argument("--threshold", help="This is the threshold value chosen by the user to determine what percentage"
                                            "of movement the centroids should be considered as converge. default=0.001",
                        default=0.001, type=float)
    parser.add_argument("--random-seed", help="This is the random seed value select by user which is used to "
                                              "choose different intial centroids. This value must be an integer and default=1.",
                        default=1, type=int)
    args = parser.parse_args()
    dataset = args.dataset
    k = args.k_val
    max_iterations = args.max_iterations
    threshold = args.threshold
    distance = args.distance
    seeds = args.random_seed
    return dataset, k, distance, max_iterations, threshold, seeds


def main():
    data_path, k, distance, max_iterations, threshold, seeds = parse_args()
    seed(seeds)
    print("The seed value now is : {}.".format(seeds))
    print('The dataset is from : {}.'.format(data_path))
    print('This is : {}-means clustering'.format(k), end=' ')
    if distance == 'Euclidean':
        print('using euclidean distance.')
    elif distance == 'Manhattan':
        print('using manhattan distance.')
    else:
        print()
        print('Error: \"{}\" is not supported.'.format(distance))
        exit(1)
    if k <= 0:
        print("K value cannot be less than 1.")
        exit(1)
    if max_iterations <= 0:
        print("max iterations cannot be less than or equal to 0.")
        exit(1)
    if threshold <= 0:
        print("converge threshold cannot be less than or equal to 0.")
        exit(1)
    print('The max #iterations is : {}.'.format(max_iterations))
    print('The converge threshold is : {}%.'.format(threshold * 100))
    dataset = generate_data(data_path)
    result, isConverge = k_means(dataset, k, distance, max_iterations, threshold)
    if isConverge == 1:
        print("This result can be converged under current situation.")
    else:
        print("This result cannot be converged under current situation.")
    for i, cluster in enumerate(result):
        positive = 0
        negative = 0
        for x, y in cluster['data_points']:
            if y == 1:
                positive += 1
            else:
                negative += 1
        print('cluster {} has positive {} and negative {}.'.format(i, positive, negative))


if __name__ == '__main__':
    main()
