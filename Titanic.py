import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())
print(df.info())
print(df.describe())

# Fill missing Age values with the median of the column
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing Cabin values with a placeholder
df['Cabin'].fillna('Unknown', inplace=True)

# Drop rows where 'Survived' is missing 
df.dropna(subset=['Survived'], inplace=True)

# Show the cleaned dataset info
print(df.info())

# Plot the distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png')
plt.close()  

# plot the distribution of fare
plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.savefig('fare_distribution.png')  
plt.close()

# Plot the count of survived vs not survived
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df)
plt.title('Count of Survived vs Not Survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('survival_count.png') 
plt.close()

# Convert categorical columns to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Feature selection
X = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

# Calculate accuracy of Decision Tree
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f'Decision Tree Accuracy: {dt_accuracy}')

# Print confusion matrix and classification report for Decision Tree
print('Confusion Matrix for Decision Tree:')
print(confusion_matrix(y_test, dt_pred))
print('Classification Report for Decision Tree:')
print(classification_report(y_test, dt_pred))

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)

# Calculate accuracy of Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_accuracy}')

# Print confusion matrix and classification report for Random Forest
print('Confusion Matrix for Random Forest:')
print(confusion_matrix(y_test, rf_pred))
print('Classification Report for Random Forest:')
print(classification_report(y_test, rf_pred))

# SVM Classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)

# Calculate accuracy of SVM
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f'SVM Accuracy: {svm_accuracy}')

# Print confusion matrix and classification report for SVM
print('Confusion Matrix for SVM:')
print(confusion_matrix(y_test, svm_pred))
print('Classification Report for SVM:')
print(classification_report(y_test, svm_pred))


# Decision Tree Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('decision_tree_confusion_matrix.png')  
plt.close()

# Random Forest Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('random_forest_confusion_matrix.png')  
plt.close()

# SVM Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('svm_confusion_matrix.png')  
plt.close()
