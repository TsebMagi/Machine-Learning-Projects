import pandas as pd
import numpy as np
from scipy.spatial import distance
import math


def create_clusters(data, size):
    # Select Random number of Clusters from data set
    return np.random.randint(17, size=(size, len(data[0, :-1])))


def update_clusters(data, clusters, centroids):
    update = False
    centroid_sum = np.zeros((len(centroids), len(data[0, :-1])))
    centroid_count = np.zeros((len(centroids)))
    # find centroid for each data element
    for x in range(len(data)):
        mdst = 999999999
        miny = 0
        for y in range(len(centroids[:, 0])):
            # calculate distance
            dst = distance.euclidean(centroids[y, :], data[x, :-1])
            # Update closest
            if dst < mdst:
                mdst = dst
                miny = y
                # Assign to Cluster based on closeness
        if clusters[x] != miny:
            update = True
            clusters[x] = miny
    # Update centroids
    for z in range(len(clusters)):
        index = int(clusters[z])
        centroid_sum[index] += data[z, :-1]
        centroid_count[index] += 1
    for c in range(len(centroids)):
        if centroid_count[c] == 0:
            centroids[c] = 0
        else:
            centroids[c] = centroid_sum[c] / centroid_count[c]
    return clusters, centroids, update


def cluster_classification(clusters, data, num_clusters):
    cluster_stats = np.zeros((num_clusters, 10), dtype='int64')
    cluster_classes = np.zeros(num_clusters, dtype='int64')
    for x in range(len(data[:, 0])):
        cluster_stats[clusters[x], data[x, -1]] += 1
    np.argmax(cluster_stats, out=cluster_classes, axis=1)
    return cluster_classes, cluster_stats


def calculate_entropy(cluster_stats):
    mean_entropy = 0
    total_instances = sum(sum(cluster_stats[y, :] for y in range(len(cluster_stats[:, 0]))))
    for x in range(len(cluster_stats[:, 0])):
        cluster_entropy = 0
        cluster_sum = sum(cluster_stats[x, :])
        total_instances += cluster_sum
        for z in range(10):
            if cluster_sum > 0:
                p = (cluster_stats[x, z] / cluster_sum)
                if p > 0:
                    cluster_entropy += p * math.log(p, 2)
        cluster_entropy *= -1
        mean_entropy += cluster_entropy * (cluster_sum / total_instances)
    return mean_entropy


def run_k_means(data, num_clusters):
    mss = 9999999999
    mse = 9999999999
    final_centers = None
    final_clusters = None
    for run in range(5):
        # Get Initial Clusters
        k_means_center = create_clusters(data, num_clusters)
        # Setup clusters
        current_clusters = np.zeros(len(data[:, 0]))
        # update clusters until no change
        done = False
        iteration = 1

        while not done:
            print("Iteration: ", iteration)
            current_clusters, k_means_center, change = update_clusters(data, current_clusters, k_means_center)
            done = not change
            iteration += 1

        # calculate the average MSE
        distances = np.zeros((num_clusters, 2))

        for x in range(len(current_clusters)):
            index = int(current_clusters[x])
            distances[index, 0] += (distance.euclidean(data[x, :-1], k_means_center[index])) ** 2
            distances[index, 1] += 1
        average_mse = 0

        for x in range(len(distances[:, 0])):
            denominator = distances[x, 1]
            numerator = distances[x, 0]
            if denominator != 0:
                average_mse += ((numerator / denominator) ** 2)
        average_mse = average_mse / num_clusters

        if average_mse < mse:
            mse = average_mse
            final_centers = k_means_center
            final_clusters = current_clusters
            separation = 0
            for i in final_centers:
                for j in final_centers:
                    separation += (distance.euclidean(i, j)) ** 2
            mss = separation / (num_clusters * (num_clusters - 1) / 2)
    k_means_classes, stats = cluster_classification(final_clusters, data, num_clusters)
    entropy = calculate_entropy(stats)
    print("Mean Square Error: ", mse)
    print("Mean Square Separation: ", mss)
    print("Mean Entropy: ", entropy)
    print("classes:", k_means_classes)
    return final_centers, k_means_classes


def calculate_accuracy(k_means_centers, k_means_classes, accuracy_test_data):
    c_matrix = np.zeros((10, 10))
    num_total = 0
    num_correct = 0
    for item in accuracy_test_data:
        min_dist = 999999999
        index = 0
        for x in range(len(k_means_centers)):
            dist = distance.euclidean(item[:-1], k_means_centers[x])
            if dist < min_dist:
                min_dist = dist
                index = x
        classification = int(k_means_classes[index])
        if classification == int(item[-1]):
            num_correct += 1
        c_matrix[classification, int(item[-1])] += 1
        num_total += 1
    print("Accuracy: ", (num_correct / num_total))
    print(c_matrix)


if __name__ == "__main__":
    # load data
    train_data = pd.read_csv("optdigits-train.txt", sep=',', dtype=np.float64, header=None).as_matrix()
    test_data = pd.read_csv("optdigits-test.txt", sep=',', dtype=np.float64, header=None).as_matrix()
    print("Experiment 1")
    centers, classes = run_k_means(train_data, 10)
    calculate_accuracy(centers, classes, test_data)
    for c in centers:
        print(c)
    print("Experiment 2")
    centers2, classes2 = run_k_means(train_data, 30)
    calculate_accuracy(centers2, classes2, test_data)
    for c in centers2:
        print(c)
