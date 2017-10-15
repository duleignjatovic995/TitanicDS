from matplotlib import pyplot as plt
import pandas as pd

# Read the dataset from CSV file
data = pd.read_csv('train.csv')

# DESCRIPTIVE STATISTICS

print(data.shape)  # Print number of examples and features (rows and columns)
print(data.head())  # Print first few examples of our dataset

print("Descriptive statistics before filling missing age values: ")
print(data.describe())  # Print descriptive statistics of our dataset

# We see from count variable that there is 177 values missing
# from age feature. We are going to replace them with median value
# which is more robust than mean in this case.
data['Age'].fillna(data['Age'].median(), inplace=True)

print("Descriptive statistics after filling missing age values: ")
print(data.describe())  # Descriptive statistics after substituting missing age values.


# VISUALISATION - plotting features against target (Survived) variable

# Now visualise survival based on gender. // w and c first!
survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
dead_sex = data[data['Survived'] == 0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15, 8), color=['r', 'c'])
plt.title('Survival based on gender')
plt.ylabel('Number of passengers')
plt.grid(True)
# plt.show()  # Only one statement at the end of file is needed


# Correlation of survival with age variable
figure = plt.figure(figsize=(15, 8))
figure.canvas.set_window_title('Survival based on age')

plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']],
         stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'], edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.title('Survival based on age')
plt.legend()
plt.grid(True)
# plt.show()
# So the passengers younger than 10 are more likely to survive!

# Analyze fare tickets and correlation with survival higher chance of death for cheaper tickets
figure = plt.figure(figsize=(15, 8))
figure.canvas.set_window_title('Survival based on ticket fare')
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'],
         bins=30, label=['Survived', 'Dead'], edgecolor='black')
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.title('Survival based on ticked fare')
plt.legend()
plt.grid(True)
# plt.show()
# Looks like the higher chance of death is for passengers with cheaper tickets


# Analyze age/fare situations
figure = plt.figure(figsize=(15, 8))
figure.canvas.set_window_title('Survival based on ticket fare and age')

ax = plt.subplot()
ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], c='green', s=40, edgecolor='black')
ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], c='red', s=40, edgecolor='black')
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.set_title('Survival based on ticket fare and age')
ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper right', fontsize=15)
# plt.show()
# A distinct cluster of dead passengers could be recognized at low fare and age between 15 and 40.


# Ticket fare correlates with the Pclass (Passenger class)
figure = plt.figure(figsize=(15, 8))
figure.canvas.set_window_title('Survival based on average fare')

ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(15, 8), ax=ax)
ax.set_title('Survival based on average fare')
# plt.show()


# Embarkation against survival
survived_embark = data[data['Survived'] == 1]['Embarked'].value_counts()
dead_embark = data[data['Survived'] == 0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark, dead_embark])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15, 8), color=['c', 'g', 'orange'])
plt.title('Survival based on embarkation')

plt.show()
# Here we se no distinct correlation
