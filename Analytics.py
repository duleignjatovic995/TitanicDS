import pandas as pd
import util
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# FEATURE ENGINEERING

# For feature engineering we are going to merge train and test set.
# The reason we are doing it is because test set doesn't have Survived
# target feature, and we want to sync our train and test set
# with all newly added features.

# PREPROCESS PIPELINE

# Combine train and test set
combined0 = util.get_combined_data()
# print(combined0.shape)
# print(combined0.head())

# Creating new feature based on title in Name value
combined1 = util.get_titles(combined0)
# print(combined1.head())  # We can see newly added feature as the last column


# We've seen that there is 177 missing values in age variable.
# Simply filling it with mean or median isn't the best possible solution.

# Group dataset by Sex, Pclass and Title and
# fill the missing values based on the groups mean
combined2 = util.process_age(combined1)
# print(combined2.head())


# Create dummy variables based on titles and drop alredy used names variable
combined3 = util.process_names(combined2)
# print(combined3.head())


# Replace missing fare values by mean
combined4 = util.process_fares(combined3)
# print(combined4.head())


# Replace missing Embarked value with most frequent
combined5 = util.process_embarked(combined4)
# print(combined5.head())


# Create dummy variable with Cabin feature
combined6 = util.process_cabin(combined5)
# print(combined6.head())


# 1 if male 0 if female
combined7 = util.process_sex_d(combined6)
# print(combined7.head())


# Create dummy variables based on passenger class
combined8 = util.process_pclass(combined7)
# print(combined8.head())


# Create dummy variables with ticket prefixes
combined9 = util.process_ticket(combined8)
# print(combined9.head())


# Create new variables based on family size
combined = util.process_family(combined9)
# print(combined.shape)
# print(combined.info())
# print(combined.head())


# Recover dataset
train0 = pd.read_csv('train.csv')

targets = train0.Survived  # Survived
train = combined.head(891)  # Initial training set
test = combined.iloc[891:]  # Initial test set


# SCALING DATA
scaler = StandardScaler()
scaler.fit(train)
train_data = scaler.transform(train)
# Restore DataFrame
train = pd.DataFrame(train_data, index=train.index, columns=train.columns)


# FEATURE SELECTION

# Tree based models can be used to extract feature importance
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
# Plot feature importance
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

reduction_model = SelectFromModel(clf, prefit=True)
train_reduced = reduction_model.transform(train_data)
# print(train_reduced.shape)
test_reduced = reduction_model.transform(test)
# print(test_reduced.shape)


# Selecting top 20 features with PCA // Doesn't actually helps a lot
# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components=2, kernel='precomputed', fit_inverse_transform=True, gamma=10)
# a = kpca.fit_transform(train)
# a = kpca.inverse_transform(a)


# DIMENSIONALITY REDUCTION FOR PLOTTING
from sklearn.decomposition import PCA

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
pca = PCA(n_components=2)
x_pca = pca.fit_transform(train_reduced)
# x_pca = pca.inverse_transform(x_pca)
reds = targets == 0
blues = targets == 1
ax1.scatter(x_pca[reds, 0], x_pca[reds, 1], c="red", s=20, edgecolor='k')
ax1.scatter(x_pca[blues, 0], x_pca[blues, 1], c="blue", s=20, edgecolor='k')
ax1.set_title("PCA Visualization")
ax1.set_xlabel("1st principal component")
ax1.set_ylabel("2nd component")


# CLUSTERING
from sklearn.cluster import SpectralClustering as Clust
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# Plot cluster predictions
# y_pred = Clust(n_clusters=11, random_state=42).fit_predict(x_pca)
y_pred = Clust().fit_predict(x_pca)
ax2.scatter(x_pca[:, 0], x_pca[:, 1], c=y_pred, s=20)
ax2.set_title("Cluster predictions")
ax2.set_xlabel("1st principal component")
ax2.set_ylabel("2nd component")


# k-means: determine optimal k using ELBOW method // k_optimal = 5 | 8 | 12
distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(train_reduced)
    kmeanModel.fit(train_reduced)
    distortions.append(
        sum(np.min(cdist(train_reduced, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / train_reduced.shape[0]
    )
# Plot the elbow
figure = plt.figure(figsize=(15, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')


# MODELING AND TUNING

# Tuning Random Forests with grid search
search = False
# If we want to continue a grid search set above variable to True
if search is True:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train_reduced, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else:
    # After doing grid search best parameters came up like this
    parameters = {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 50,
                  'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train_reduced, targets)


# Print mean cv score
print("Mean score: ", util.compute_score(model, train_reduced, targets, scoring='accuracy'))


plt.show()  # For showing plots
