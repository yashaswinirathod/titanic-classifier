# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')
# Data preprocessing
titanic_data = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) # Drop unnecessary columns
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1}) # Convert 'Sex' to numeric
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) # Convert 'Embarked' to numeric
titanic_data = titanic_data.fillna(titanic_data.mean()) # Fill missing values with mean
# Split the data into features and target
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
# Make predictions
y_pred = rf_classifier.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
