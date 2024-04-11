import pandas as pd

# Load the dataset
data = pd.read_csv('Occupancy_Estimation.csv')

# Drop 'Date' and 'Time' columns
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Split data into features and target
X = data.drop('Room_Occupancy_Count', axis=1)
y = data['Room_Occupancy_Count']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

import matplotlib.pyplot as plt

# Plotting the predicted room occupancy count
plt.figure(figsize=(12, 6))
plt.bar(range(len(y_test)), y_test, color='b', alpha=0.5, label='Actual')
plt.bar(range(len(y_pred)), y_pred, color='r', alpha=0.5, label='Predicted')
plt.xlabel('Samples')
plt.ylabel('Room Occupancy Count')
plt.title('Actual vs Predicted Room Occupancy Count')
plt.legend()
plt.show()
