import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn import ensemble

# loading the dataset
data = pandas.read_csv("train.csv")
# print(data.shape)
# print(data.info())
# print(data.head())


# dropping the values that we cannot make use of right now
cols_to_drop = ['Name', 'Ticket', 'Cabin']
data = data.drop(cols_to_drop, axis=1)
# df.info()


# we will now remove the rows which have unknown data
data = data.dropna()
# print(data)


dummies = []
cat_cols = ['Sex', 'Pclass', 'Embarked']
for c in cat_cols:
    dummies.append(pandas.get_dummies(data[c]))

titanic_dummies = pandas.concat(dummies, axis=1)
# print(titanic_dummies)

# concat dummy variables to previous data and drop original columns
data = pandas.concat((data, titanic_dummies), axis=1).drop(['Sex', 'Pclass', 'Embarked'], axis=1)
print("After adding dummies: ")
print(data.info())
print(data.head())

# Interpolate NaN values from dataset
data['Age'] = data['Age'].interpolate()
# print(df['Age'])


# NUMPY PART
X = data.values
y = data['Survived'].values

X = numpy.delete(X, 1, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = ensemble.RandomForestClassifier()
clf.fit(X_train, y_train)
print("GBoost: ", clf.score(X_test, y_test))
