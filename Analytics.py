import pandas as pd
import util
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# FEATURE ENGINEERING

# For feature engineering we are going to merge train and test set.
# The reason we are doing it is because test set doesn't have Survived
# target feature, and we want to sync our train and test set
# with all newly added features.

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


# FEATURE SELECTION

# Tree based models can be used to
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

# Plot feature importance
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))
plt.show()


model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
# print(train_reduced.shape)


test_reduced = model.transform(test)
# print(test_reduced.shape)


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

    grid_search.fit(train, targets)
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
print("Mean score: ", util.compute_score(model, train, targets, scoring='accuracy'))
