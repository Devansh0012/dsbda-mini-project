from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the data from the POST request.
    #data = request.get_json(force=True)
    S1_Temp = float(request.form['S1_Temp'])
    S2_Temp = float(request.form['S2_Temp'])
    S3_Temp = float(request.form['S3_Temp'])
    S4_Temp = float(request.form['S4_Temp'])
    S1_Light = float(request.form['S1_Light'])
    S2_Light = float(request.form['S2_Light'])
    S3_Light = float(request.form['S3_Light'])
    S4_Light = float(request.form['S4_Light'])
    S1_Sound = float(request.form['S1_Sound'])
    S2_Sound = float(request.form['S2_Sound'])
    S3_Sound = float(request.form['S3_Sound'])
    S4_Sound = float(request.form['S4_Sound'])
    S5_CO2 = float(request.form['S5_CO2'])
    S5_CO2_Slope = float(request.form['S5_CO2_Slope'])
    S6_PIR = float(request.form['S6_PIR'])
    S7_PIR = float(request.form['S7_PIR'])
    # Add code to get other sensor readings here

    # Reshape the input data
    sensor_readings = [float(request.form[key]) for key in ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 
                                                             'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 
                                                             'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 
                                                             'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']]
    # Reshape the input data into a 2D array
    data = np.array(sensor_readings).reshape(-1, 1)
    
    # Preprocess the input data
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(data)
    
    return str(predictions)

if __name__ == '__main__':
    app.run(debug=True)
    