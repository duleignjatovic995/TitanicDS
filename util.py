import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


def status(feature):
    """
    Function used to print successful feature processing
    
    :param feature: Feature from dataset
    """
    print('Processing ', feature, ': ok')


def get_combined_data():
    """
    Function for merging train and test set
    
    :return: Combined train and test dataframe.
    """
    # reading train data
    train = pd.read_csv('train.csv')

    # reading test data
    test = pd.read_csv('test.csv')

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined_data = train.append(test)
    combined_data.reset_index(inplace=True)
    combined_data.drop(['index', 'PassengerId'], inplace=True, axis=1)

    return combined_data


def get_titles(combined_data):
    """
    Function for getting titles from passenger names
    and adding new feature (Title) to our dataset.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # we extract the title from each name
    combined_data['Title'] = combined_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    title_dictionary = {
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
    combined_data['Title'] = combined_data.Title.map(title_dictionary)
    return combined_data


def process_age(combined_data):
    """
    Function for processing and filling missing age values.
    
    :param combined_data: Dataset
    :return Improved dataset
    """

    # a function that fills the missing values of the Age variable

    def fill_ages(row, grouped_median):
        """
        Helper function for processing missing age values.
        :param row: Row number (Data Example)
        :param grouped_median: Value for filling missing age values. 
        """
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

    # Processing age and finding better solution than the mean
    # To prevent data leakage, we preform operations separately.

    # For the train set
    grouped_train = combined_data.head(891).groupby(['Sex', 'Pclass', 'Title'])
    grouped_median_train = grouped_train.median()

    # And for the test set
    grouped_test = combined_data.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
    grouped_median_test = grouped_test.median()

    combined_data.head(891).Age = combined_data.head(891).apply(
        lambda r: fill_ages(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'], axis=1
    )
    print("pre ovoga")
    combined_data.iloc[891:].Age = combined_data.iloc[891:].apply(
        lambda r: fill_ages(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1
    )

    status('age')
    return combined_data


# replaces one missing Fare value by the mean
def process_names(combined_data):
    """
    Function for preprocessing Title feature in passengers names and
    based on them adding new variables to dataset.
    
    :param combined_data: Dataset
    :return: Improved dataset
    """
    # we clean the Name variable
    combined_data.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined_data['Title'], prefix='Title')
    combined_data = pd.concat([combined_data, titles_dummies], axis=1)

    # removing the title variable since it's no longer useful
    combined_data.drop('Title', axis=1, inplace=True)

    status('names')
    return combined_data


def process_fares(combined_data):
    """
    Function for processing fares and replacing missing data with
    mean value.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # there's one missing fare value - replacing it with the mean.
    combined_data.head(891).Fare.fillna(combined_data.head(891).Fare.mean(), inplace=True)

    # Do it separately for test set
    combined_data.iloc[891:].Fare.fillna(combined_data.iloc[891:].Fare.mean(), inplace=True)

    status('fare')
    return combined_data


def process_embarked(combined_data):
    """
    Function for processing embarkation and creating
    dummy variables based on it.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # two missing embarked values - filling them with the most frequent one (S)
    combined_data.head(891).Embarked.fillna('S', inplace=True)
    combined_data.iloc[891:].Embarked.fillna('S', inplace=True)

    # dummy encoding
    embarked_dummies = pd.get_dummies(combined_data['Embarked'], prefix='Embarked')
    combined_data = pd.concat([combined_data, embarked_dummies], axis=1)
    combined_data.drop('Embarked', axis=1, inplace=True)

    status('embarked')
    return combined_data


def process_cabin(combined_data):
    """
    Function for processing Cabin feature and creating
    dummy variables based on it.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # replacing missing cabins with U (for Uknown)
    combined_data.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined_data['Cabin'] = combined_data['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined_data['Cabin'], prefix='Cabin')

    combined_data = pd.concat([combined_data, cabin_dummies], axis=1)

    combined_data.drop('Cabin', axis=1, inplace=True)

    status('cabin')
    return combined_data


def process_sex(combined_data):
    """
    Function for changing string values from
    sex variable to 1 for male and 0 for female.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # mapping string values to numerical one
    combined_data['Sex'] = combined_data['Sex'].map({'male': 1, 'female': 0})

    status('sex')
    return combined_data


def process_sex_d(combined_data):
    """
    Function for processing sex feature and creating
    dummy variables based on it.
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # encoding into 2 categories:
    sex_dummies = pd.get_dummies(combined_data['Sex'])

    # adding dummy variables
    combined_data = pd.concat([combined_data, sex_dummies], axis=1)

    # removing "Sex" since it's no loner needed
    combined_data.drop('Sex', axis=1, inplace=True)

    status('pclass')
    return combined_data


def process_pclass(combined_data):
    """
    Function for processing Pclass feature and creating
    dummy variables based on it.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined_data['Pclass'], prefix="Pclass")

    # adding dummy variables
    combined_data = pd.concat([combined_data, pclass_dummies], axis=1)

    # removing "Pclass" since it's no loner needed
    combined_data.drop('Pclass', axis=1, inplace=True)

    status('pclass')
    return combined_data


def process_ticket(combined_data):
    """
    Function for processing ticket variable by extracting it's prefix
    and creating dummy variables based on that prefix.
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """

    def clean_ticket(ticket):
        """
        Helper function for processing ticket variable by firstly
        extracting ticket prefix, if it fails in extracting it returns XXX.
        
        :param ticket: Ticket entry
        :return: Prefix code from ticket
        """
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
    combined_data['Ticket'] = combined_data['Ticket'].map(clean_ticket)
    tickets_dummies = pd.get_dummies(combined_data['Ticket'], prefix='Ticket')
    combined_data = pd.concat([combined_data, tickets_dummies], axis=1)
    combined_data.drop('Ticket', inplace=True, axis=1)

    status('ticket')
    return combined_data


def process_family(combined_data):
    """
    Function for processing Parch and SibSp features and 
    creating four new features.
    1. FamilySize: number of siblings and parents/children.
    2. Singleton: 1 if family size is 1, 0 otherwise
    3. SmallFamily: 1 if family size is between 2 and 4(including 2 and 4), 0 otherwise
    4. LargeFamily: 1 if family size is larger than 4, 0 otherwise. 
    
    :param combined_data: Dataset
    :return: Improved Dataset
    """
    # introducing a new feature : the size of families (including the passenger)
    combined_data['FamilySize'] = combined_data['Parch'] + combined_data['SibSp'] + 1

    # introducing other features based on the family size
    combined_data['Singleton'] = combined_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined_data['SmallFamily'] = combined_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined_data['LargeFamily'] = combined_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')
    return combined_data


def compute_score(clf, X, y, scoring='accuracy', cv=5):
    """
    Function for computing mean k-fold cross validation score.
    
    :param clf: Predictor
    :param X: Feature data
    :param y: Target data
    :param scoring: scoring type -> string
    :param cv: number k for k-fold cross validation
    :return: 
    """
    x_val = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return np.mean(x_val)
