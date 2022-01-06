import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from tabulate import tabulate

# Loading data from file
dataset = np.genfromtxt("hepatitis.dat", delimiter=",")
dataset = dataset.astype(np.int32)

data = dataset[1:, 0:-1]
classes = dataset[1:, -1]

# Classifiers dictionary (3 k values and 2 metrics)
classifiers = {
    "euclidean_3": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
    "euclidean_5": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "euclidean_9": KNeighborsClassifier(n_neighbors=9, metric="euclidean"),
    "manhattan_3": KNeighborsClassifier(n_neighbors=3, metric="manhattan"),
    "manhattan_5": KNeighborsClassifier(n_neighbors=5, metric="manhattan"),
    "manhattan_9": KNeighborsClassifier(n_neighbors=9, metric="manhattan")
}

# Features dictionary (7 best form Chi2 ranking)
features = {
    '1 feat': (data[:, [17]]),  # ProTime
    '2 feat': (data[:, [17, 14]]),  # ProTime, AlkPhosphate
    '3 feat': (data[:, [17, 14, 15]]),  # ProTime, AlkPhosphate, Sgot
    '4 feat': (data[:, [17, 14, 15, 0]]),  # ProTime, AlkPhosphate, Sgot, Age
    '5 feat': (data[:, [17, 14, 15, 0, 13]]),  # ProTime, AlkPhosphate, Sgot, Age, Bilirubin real
    '6 feat': (data[:, [17, 14, 15, 0, 13, 18]]),  # ProTime, AlkPhosphate, Sgot, Age, Bilirubin real, Histology
    '7 feat': (data[:, [17, 14, 15, 0, 13, 18, 16]])
    # ProTime, AlkPhosphate, Sgot, Age, Bilirubin real, Histology, AlbuMin real
}

# Repeated k-Fold Cross Validation
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)  # Stratified - unbalanced data
scores = np.zeros((len(features), len(classifiers), 10))

for feat_count, feat_name in enumerate(features):  # Feature loop (7 times)
    feat = features[feat_name]
    for fold_count, (train_index, test_index) in enumerate(rskf.split(feat, classes)):  # Fold loop (10 times)
        for clf_count, clf_name in enumerate(classifiers):  # Classifier loop (6 times)
            clf = classifiers[clf_name]
            clf.fit(feat[train_index], classes[train_index])
            predict = clf.predict(feat[test_index])
            scores[feat_count, clf_count, fold_count] = accuracy_score(classes[test_index], predict)

#  Arithmetic mean and standard deviation
mean = np.mean(scores, axis=2)
std = np.std(scores, axis=2)

for feat_count, feat_name in enumerate(features):  # Feature loop (7 times)
    for clf_count, clf_name in enumerate(classifiers):  # Classifier loop (6 times)
        print("Feature: %s, Classifier: %s, Mean: %.3f, Standard deviation: %.3f" % (
            feat_name, clf_name, mean[feat_count][clf_count], std[feat_count][clf_count]))

# Student's t-test
headers = ["euclidean_3", "euclidean_5", "euclidean_9", "manhattan_3", "manhattan_5", "manhattan_9"]
names_column = np.array(
    [["euclidean_3"], ["euclidean_5"], ["euclidean_9"], ["manhattan_3"], ["manhattan_5"], ["manhattan_9"]])

alpha = 0.05

t_statistic = np.zeros((len(classifiers), len(classifiers)))

p_value = np.zeros((len(classifiers), len(classifiers)))

advantage = np.zeros((len(classifiers), len(classifiers)))

significance = np.zeros((len(classifiers), len(classifiers)))

for f in range(len(features)):
    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[f][i], scores[f][j])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")

    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)

    significance[p_value <= alpha] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)

    print("\n\n\n")
    print(f"\n{f+1} Feature(s)")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    print("\nAdvantage:\n", advantage_table)
    print(f"\nStatistical significance (alpha = {alpha}):\n", significance_table)
    print("\nStatistically significantly better:\n", stat_better_table)
