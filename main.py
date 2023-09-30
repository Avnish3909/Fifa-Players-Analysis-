import pandas as pd
import numpy as np
import shc as shc
import ydata_profiling
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import argsort
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, f1_score, \
    precision_score, mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from ydata_profiling import ProfileReport
import warnings
from gap_statistic import OptimalK
from scipy.cluster import hierarchy as shc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/dell/Downloads/Fifa.csv")

df.drop(columns=["Unnamed: 0"], inplace=True)

print(df.head().to_string())
print(df.columns)

df_1 = df[['Name', 'Age', 'Overall',
           'Potential', 'Value', 'Wage', 'International Reputation',
           'Skill Moves', 'Club', 'Nationality', 'Crossing',
           'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
           'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
           'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
           'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
           'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
           'Marking', 'StandingTackle', 'SlidingTackle']]
print(df_1.head().to_string())

df_0 = df_1.sample(100)
print(df_0.head().to_string())

df_1 = df_0.drop(columns=["Name", "Value", "Wage", "Club", "Nationality", "Overall", "Potential"])

print(df_1.head().to_string())
min = MinMaxScaler()
X_min = min.fit_transform(df_1)
print(X_min)
print(X_min.shape)
print(df_0.columns)

simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X_min = simple_imputer.fit_transform(X_min)

pca = PCA(n_components=32)
pca.fit(X_min)
print(pca)
print(pca.explained_variance_ratio_)

pca = PCA().fit(X_min)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
# plt.show()

pca = PCA(n_components=10)
pca.fit(X_min)
print(pca)
score_pca = pca.transform(X_min)
print(pca.explained_variance_ratio_[:10].sum())

num_cluster = range(2, 25)
error = []
for num in num_cluster:
    clusters = KMeans(n_clusters=num)
    clusters.fit(score_pca)
    error.append(clusters.inertia_ / 100)
temp = pd.DataFrame({"num_cluster": num_cluster, "error": error})
print(temp)

plt.figure(figsize=(10, 6))
plt.plot(temp["num_cluster"], temp["error"])
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Error")
# plt.show()

clusters1 = KMeans(n_clusters=6)
clusters1.fit(score_pca)
print(clusters1.labels_)

plt.figure(figsize=(10, 6))
plt.scatter(score_pca[:, 0], score_pca[:, 1], c=clusters1.labels_)
plt.grid(True)


# plt.show()

def currencyconverter(amount):
    n_amount = []
    for i in amount:
        i = i.replace('€', '')
        abbr = i[-1]
        i = i[:-1]
        if abbr == "M":
            i = float(i) * 1000000
        elif abbr == "K":
            i = float(i) * 1000
        elif abbr == "B":
            i = float(i) * 1000000000
        elif abbr == "T":
            i = float(i) * 1000000000000
        else:
            i = 0
        n_amount.append(i)
    return n_amount


df_0["Value"] = currencyconverter(df_0["Value"])
df_0["Wage"] = currencyconverter(df_0["Wage"])

Players = df_0["Name"]
wage = df_0["Wage"] / 1000
value = df_0["Value"] / 100000
Overall = df_0["Overall"]
Age = df_0["Age"]

dff = pd.DataFrame(
    {"clusters": clusters1.labels_, "Name": Players, "Overall": Overall, "Age": Age, "Wages in thousands": wage,
     "Transfer Value in millions": value})
sorted_dff=dff.sort_values("clusters")
print(sorted_dff.to_string())
print(dff.groupby("clusters").describe().transpose().to_string())













