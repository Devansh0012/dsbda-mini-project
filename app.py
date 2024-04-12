from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Occupancy_Estimation.csv')

# Drop 'Date' and 'Time' columns
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Calculate average temperature, light, and sound values for each room occupancy count
avg_data = data.groupby('Room_Occupancy_Count').mean()

# Create a plot of average values against room occupancy count
plt.figure(figsize=(10, 6))
plt.plot(avg_data.index, avg_data['S1_Temp'], label='Average Temperature')
plt.plot(avg_data.index, avg_data['S1_Light'], label='Average Light')
plt.plot(avg_data.index, avg_data['S1_Sound'], label='Average Sound')
plt.xlabel('Room Occupancy Count')
plt.ylabel('Average Value')
plt.title('Average Temperature, Light, and Sound vs Room Occupancy Count')
plt.legend()
plt.grid(True)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
buffer = b''.join(buf)
plot_data = base64.b64encode(buffer).decode('utf8')
plt.close()

# Split data into features and target
X = data.drop('Room_Occupancy_Count', axis=1)
y = data['Room_Occupancy_Count']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create classifiers
rf_model = RandomForestClassifier()
svm_model = SVC()
dt_model = DecisionTreeClassifier()

# Train the models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Calculate accuracies
rf_accuracy = accuracy_score(y_test, rf_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get sensor values from the form
        sensor_values = [float(request.form['S1_Temp']),
                         float(request.form['S2_Temp']),
                         float(request.form['S3_Temp']),
                         float(request.form['S4_Temp']),
                         float(request.form['S1_Light']),
                         float(request.form['S2_Light']),
                         float(request.form['S3_Light']),
                         float(request.form['S4_Light']),
                         float(request.form['S1_Sound']),
                         float(request.form['S2_Sound']),
                         float(request.form['S3_Sound']),
                         float(request.form['S4_Sound']),
                         float(request.form['S5_CO2']),
                         float(request.form['S5_CO2_Slope']),
                         float(request.form['S6_PIR']),
                         float(request.form['S7_PIR'])]

        # Make predictions for each model
        rf_prediction = rf_model.predict([sensor_values])[0]
        svm_prediction = svm_model.predict([sensor_values])[0]
        dt_prediction = dt_model.predict([sensor_values])[0]

        prediction = {
            'Random Forest': rf_prediction,
            'SVM': svm_prediction,
            'Decision Tree': dt_prediction
        }

    return render_template('index.html', prediction=prediction, rf_accuracy=rf_accuracy, svm_accuracy=svm_accuracy, dt_accuracy=dt_accuracy, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
