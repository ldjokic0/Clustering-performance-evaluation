import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def plot_matrix(matrix, annotations, fig_size=(12, 12), label_size=15, title=''):
    plt.figure(figsize=fig_size, dpi=600)
    plt.title(title)
    sns.heatmap(matrix, yticklabels=annotations, cmap="Blues", annot=True, annot_kws={"size": label_size})
    plt.tick_params(axis='x', labelsize=label_size)
    plt.tick_params(axis='y', labelsize=label_size)


def calc_label_occurence(df):

    # Find unique ground truth labels
    unique_ground_truth_labels = df.annotation.unique().tolist()
    # Sum the total number of occurrences of each label in ground truth labels
    n_label_occurence = {}
    for label in unique_ground_truth_labels:
        n_label_occurence[label] = df.annotation.str.count(label).sum()

    return n_label_occurence


def get_clusters_ground_truth(df, column1, column2):

    """Column1 is the name of the first, and
     column2 is the name of the second column used for geting coresponding ground truth labels."""

    # Initialize empty lists
    clusters, ground_truth = [], []

    # Find unique clusters and their corresponding ground truth labels
    for i in range(0, len(df[column1].unique())):
        clusters.append(df.loc[df[column1] == i, column1].tolist())
        ground_truth.append(df.loc[df[column1] == i, column2].tolist())

    return clusters, ground_truth


def calc_prec_labels_in_clusters(ground_truths, n_label_occurence, comparison_type):

    prec_labels_in_clusters = {}

    # For each cluster calculate percentage based on comaprison type
    for count, ground_truth in enumerate(ground_truths):

        prec_labels = []
        for label in n_label_occurence.keys():
            if label in ground_truth:
                if comparison_type == "clusters":
                    prec_labels.append(ground_truth.count(label) / len(ground_truth))
                elif comparison_type == "dataframe":
                    prec_labels.append(ground_truth.count(label) / n_label_occurence[label])
            else:
                prec_labels.append(0)

        prec_labels_in_clusters[count] = prec_labels

    return prec_labels_in_clusters


def cosine_similarity(c1, c2):
    labels = set(c1).union(c2)
    dotprod = sum(c1.get(n, 0) * c2.get(n, 0) for n in labels)
    magA = math.sqrt(sum(c1.get(n, 0)**2 for n in labels))
    magB = math.sqrt(sum(c2.get(n, 0)**2 for n in labels))
    return dotprod / (magA * magB)


def cosine_similiarity_matrix(ground_truths):
    # Create counters for every cluster
    counters = []
    for truth in ground_truths:
        counters.append(Counter(truth))

    matrix = np.zeros((len(ground_truths), len(ground_truths)))

    for n, counter1 in enumerate(counters):
        for k, counter2 in enumerate(counters):
            matrix[n][k] = cosine_similarity(counter1, counter2)

    return np.around(matrix, 2)


def make_matrix(df, comparison_type="clusters", cluster_column="clusters"):

    """There are two comparison types (default = "clusters"):
    - "clusters" is used when calculating precentage of ground truth labels within cluster
    - "dataframe" is used when calculating precentage of the total number of each ground truth label witihin cluster
    - "cosine" is used when calculating cosine similiarity between ground truth clusters coresponding to each initial cluster

      Also, there is a choice of column for geting cluster labels, default is "cluster",
      but other columns from dataframe can be specified."""

    # Get clustering labels from clsuter_column and ground truth labels from column annotations
    clusters, ground_truths = get_clusters_ground_truth(df, cluster_column, "annotation")

    if comparison_type == "cosine":
        return cosine_similiarity_matrix(ground_truths)

    # Get a dictionary where keys are unique labels
    # and values are total number of occurences of each label in dataframe
    n_label_occurence = calc_label_occurence(df)

    # For every cluster calculate precentage of occurences of each label from ground truth
    # based on comaprison type
    prec_labels_in_clusters = calc_prec_labels_in_clusters(ground_truths, n_label_occurence, comparison_type)

    matrix = np.zeros((len(n_label_occurence), len(clusters)))

    for key, value in prec_labels_in_clusters.items():
        for i in range(len(n_label_occurence)):
            matrix[i][key] = value[i]

    # Rounding is performed to produce better visualization in plots later
    return np.around(matrix, 2)


def group_similiar_clusters(matrix, similiarity=0.85):

    """ Function used to create dictionary by using cosine similiarity matrix to group similiar clusters.
        Keys represent clusters and values are lists with values greather than paramater similiarity. """

    d = {}
    for i in range(len(matrix)):
        row = matrix[i]
        res = [index for index, value in enumerate(row) if value > similiarity]

        # Removes index of current cluster because their similiarity is always equal to 1
        res.remove(i)
        if res == [] or any(x in d.keys() for x in res):
            continue
        d[i] = res

    return d


def remapping(matrix):

    """ Function which mapps each cluster to the ground truth cluster
    based on the maximum value in each cluster column."""

    mapping = {}
    for i in range(len(matrix)):
        row = matrix[:, i]
        index_highest = [index for index, value in enumerate(row) if value == max(row)]
        mapping[i] = index_highest[0]

    return mapping


def evaluate_clustering(df, column_name):

    """ Three existing metrics for evaluation are used and their abberevation are:
        ari - Adjusted Rand index
        ami - Adjusted Mutual Information
        fmi - Fowlkes-Mallows Index """

    ari = metrics.adjusted_rand_score(df[column_name], df["ground truth"])
    ami = metrics.adjusted_mutual_info_score(df[column_name], df["ground truth"])
    fmi = metrics.fowlkes_mallows_score(df[column_name], df["ground truth"])

    print("Adjusted Rand index is: " + str(round(ari, 2)))
    print("Adjusted Mutual Information is: " + str(round(ami, 2)))
    print("Fowlkes-Mallows Index is: " + str(round(fmi, 2)))

    return ari, ami, fmi


def purity_index(clustering_ground_truth, n):

    """ Function return value from 0 to 1 which represents how "pure" each cluster is.
    It is calculated by mathematical formula:

        purity = (x_1 / total)**2 + (x_2 / total)**2 + ... + (x_i / total)**2

        x_1, x_2, ..., x_i - represents the sum of i-th label within cluster
        i = (0, 1, 2, ..., num_of_labels)
        total - represent sum of all labels/elements within cluster

    Purity is calculated for each cluster and each purity is multiplied with clusters corresponding purity weight.
    Purity weight is calculated with mathematical expression:

        purity_weight = total / n_all_elements

        total - represent sum of all labels/elements within cluster
        n_all_elements (paramater n) - represent sum of all elements in all clusters

    Clustering purity is calculated by suming up all purities in other word purities of each cluster are added together.

        clustering_purity = purity_1 + purity_2 + ... + purity_n

        n = (0, 1, 2, ..., num_of_clusters)

    When all labels/elements in cluster are same, purity will be 1.
    """

    # Makes list containing Counters which are storing all unique labels
    # and number of instances of each label in each cluster
    counters = []
    for cluster in clustering_ground_truth:
        counters.append(Counter(cluster))

    all_purities = []
    for counter in counters:

        total = sum(counter.values())
        purity = 0

        for value in counter.values():
            purity += ((value/total)**2)

        all_purities.append(purity)

        # Purity is multiplied by clusters purity weight
        all_purities[-1] *= total / n

    return sum(all_purities)
