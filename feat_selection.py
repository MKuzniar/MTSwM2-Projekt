from sklearn.feature_selection import chi2
import pandas as pd

# Loading data from file
classes = pd.read_csv('hepatitis.dat', usecols=[19])
features_names = pd.read_csv('features.dat')

data = pd.read_csv('hepatitis.dat', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
print(data)

# Chi2 feature selection
chi2, p = chi2(data, classes)
results = pd.DataFrame(chi2, columns=["Chi"])

results["Feature"] = features_names
results.sort_values(by=['Chi'], ascending=False, inplace=True)

print(f"\n {results}")
