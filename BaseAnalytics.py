import pandas

# loading the dataset

df = pandas.read_csv("train.csv")
# df.info()


# dropping the values that we cannot make use of right now
cols_to_drop = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols_to_drop, axis=1)
# df.info()


# we will now remove the rows which have unknown data
# TODO: RESOLVE MISSING VALUES!!!
df = df.dropna()
# print(df)


dummies = []
cat_cols = ['Sex', 'Pclass', 'Embarked']
for c in cat_cols:
    dummies.append(pandas.get_dummies(df[c]))

titanic_dummies = pandas.concat(dummies, axis=1)
# print(titanic_dummies)

# concat dummy variables to previous data and drop original columns
df = pandas.concat((df, titanic_dummies), axis=1).drop(['Sex', 'Pclass', 'Embarked'], axis=1)
df.info()

# Interpolate NaN values from dataset
df['Age'] = df['Age'].interpolate()
# print(df['Age'])

from sklearn.model_selection import train_test_split
import numpy

# NUMPY PART
X = df.values
y = df['Survived'].values

X = numpy.delete(X, 1, axis=1)
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM

# clf = ensemble.RandomForestClassifier()
# clf.fit(X_train, y_train)
# print("Rand For: ", clf.score(X_test, y_test))


clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)
print("GBoost: ", clf.score(X_cv, y_cv))


# clf = ensemble.AdaBoostClassifier()
# clf.fit(X_train, y_train)
# print("Ada: ", clf.score(X_test, y_test))


# clf = svm.SVC(kernel="linear", C=4)
# clf.fit(X_train, y_train)
# print("svm: ", clf.score(X_test, y_test))


# clf = MLPClassifier(hidden_layer_sizes=(272,), solver='lbfgs', nesterovs_momentum=False)
# clf.fit(X_train, y_train)
# print("NN: ", clf.score(X_test, y_test))


