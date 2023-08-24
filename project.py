import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Hackton_predicted.csv")
x = data[["country_name"]]
y = data['medal_type']

# One-hot encoding for categorical feature 'country_name'
x_encoded = pd.get_dummies(x, drop_first=True)  # Use drop_first to avoid the dummy variable trap

# Label encoding for the target variable 'medal_type'
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=0)

# Initialize the Random Forest Classifier and train it
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=147, random_state=0)
classifier.fit(x_train, y_train)

# Predict 'medal_type' for the test set
y_pred = classifier.predict(x_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)
