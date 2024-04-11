from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Occupancy_Estimation.csv')

# Drop 'Date' and 'Time' columns
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Split data into features and target
X = data.drop('Room_Occupancy_Count', axis=1)
y = data['Room_Occupancy_Count']

# Create a Random Forest Classifier
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X, y)

sensor_values = [0] * len(X.columns)  # Default values for sensor values

@app.route('/', methods=['GET', 'POST'])
def index():
    global sensor_values  # Use the global sensor_values variable

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

        # Make prediction
        prediction = rf_model.predict([sensor_values])[0]

    # Generate bar plot for visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sensor_values)), sensor_values, color='b', alpha=0.5)
    plt.xlabel('Sensor')
    plt.ylabel('Sensor Value')
    plt.title('Sensor Values')
    plt.xticks(range(len(sensor_values)), ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR'], rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    buffer = b''.join(buf)
    plot_data = base64.b64encode(buffer).decode('utf8')
    plt.close()

    return render_template('index.html', prediction=prediction, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
