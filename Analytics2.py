import pandas as pd
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

pd.options.display.max_rows = 100

# todo resi organizacija
# todo deprecated i warning
# todo bolje plotovanje

data = pd.read_csv('train.csv')
# print(data.shape)
# print(data.head())
# print(data.describe())

# We see from count variable that there is 177 values missing
# from age feature. We are going to replace them with median value
# which is more robust than mean in this case.

data['Age'].fillna(data['Age'].median(), inplace=True)


# print(data.describe())

# VISUALISATION - plotting features against target variable

# Now visualise survival based on gender. // w and c first!
# survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
# dead_sex = data[data['Survived'] == 0]['Sex'].value_counts()
#
# df = pd.DataFrame([survived_sex, dead_sex])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize=(15, 8))
# plt.show()


# Correlation of survival with age variable
# figure = plt.figure(figsize=(15, 8))
# plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'],
#          bins=30, label=['Survived', 'Dead'])
# plt.xlabel('Age')
# plt.ylabel('Number of passengers')
# plt.legend()
# plt.grid(True)
# plt.show()

# Analyze fare tickets and correlation with survival //higher chance of death for cheaper tickets
# figure = plt.figure(figsize=(15, 8))
# plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'],
#          bins=30, label=['Survived', 'Dead'], edgecolor='black')
# plt.xlabel('Fare')
# plt.ylabel('Number of passengers')
# plt.legend()
# plt.grid()
# plt.show()

# Analyze age/fare situations //we see a distinct cluster of dead passengers(low fare, age 15-40)
# plt.figure(figsize=(15, 8))
# ax = plt.subplot()
# ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], c='green', s=40, edgecolor='black')
# ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], c='red', s=40, edgecolor='black')
# ax.set_xlabel('Age')
# ax.set_ylabel('Fare')
# ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper right', fontsize=15, )
# plt.show()

# Ticket fare correlates with the Pclass
# ax = plt.subplot()
# ax.set_ylabel('Average fare')
# data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
# plt.show()

# Embarkation against survival // no distincr correlation
# survived_embark = data[data['Survived'] == 1]['Embarked'].value_counts()
# dead_embark = data[data['Survived'] == 0]['Embarked'].value_counts()
# df = pd.DataFrame([survived_embark, dead_embark])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize=(15, 8))
# plt.show()


# FEATURE ENGINEERING

def status(feature):
    print('Processing ', feature, ': ok')


def get_combined_data():
    # reading train data
    train = pd.read_csv('train.csv')

    # reading test data
    test = pd.read_csv('test.csv')

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined


combined = get_combined_data()


# print(combined.shape)
# print(combined.head())


# adding name titles in dataset
def get_titles():
    global combined

    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }

    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)


get_titles()
# print(combined.head())

# Processing age and finding better solution than the mean
# For the train set
grouped_train = combined.head(891).groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()

# And for the test set
grouped_test = combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_test = grouped_test.median()


# print(grouped_median_train)
# print(grouped_median_test)

# **REPLACING MISSING VALUES**

# todo fix warning
def process_age():
    global combined

    # a function that fills the missing values of the Age variable

    def fillAges(row, grouped_median):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

    combined.head(891).Age = combined.head(891).apply(lambda r: fillAges(r, grouped_median_train) if np.isnan(r['Age'])
    else r['Age'], axis=1)

    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r: fillAges(r, grouped_median_test) if np.isnan(r['Age'])
    else r['Age'], axis=1)

    status('age')


# Finnaly replace missing values for age
process_age()


# combined.info()

# replaces one missing Fare value by the mean
def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)

    status('names')


process_names()


def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)

    status('fare')


process_fares()


def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)

    status('embarked')


process_embarked()


def process_cabin():
    global combined

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')


process_cabin()


def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

    status('sex')


process_sex()


def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variables
    combined = pd.concat([combined, pclass_dummies], axis=1)

    # removing "Pclass"

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')


def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variables
    combined = pd.concat([combined, pclass_dummies], axis=1)

    # removing "Pclass"

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')


process_pclass()


def process_ticket():
    global combined

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('ticket')


process_ticket()


# print(combined.head())


def process_family():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')


process_family()
# print(combined.shape)

# drop the useless feature
combined.drop('PassengerId', inplace=True, axis=1)


# MODELING

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)


def recover_train_test_target():
    global combined

    train0 = pd.read_csv('train.csv')

    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]

    return train, test, targets


train, test, targets = recover_train_test_target()

# FEATURE SELECTION


# Tree-based estimators can be used to compute feature importances, which in turn
# can be used to discard irrelevant features.
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))
# plt.show()

# Transforming dataset
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)
test_reduced = model.transform(test)
print(test_reduced.shape)

# Tuning Random Forests
run_gs = False

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [1, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

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
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)

print(compute_score(model, train, targets, scoring='accuracy'))
