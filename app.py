from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)

# Load and train model
def train_model():
    data = pd.read_csv('data/pregnancy_data.csv')
    X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
    y = data['RiskLevel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)


    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")



    # Save model
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load or train model
try:
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Prepare input data
    input_data = [[
        float(data['age']),
        float(data['systolic']),
        float(data['diastolic']),
        float(data['bs']),
        float(data['bodyTemp']),
        float(data['heartRate'])
    ]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Generate recommendations based on risk level
    recommendations = get_recommendations(prediction, data)
    
    return jsonify({
        'prediction': prediction,
        'recommendations': recommendations
    })

def get_recommendations(risk_level, data):
    recommendations = []
    
    if risk_level == 'high risk':
        recommendations = [
            "Immediate medical consultation required",
            "Regular blood pressure monitoring",
            "Strict diet control",
            "Complete bed rest advised"
        ]
    elif risk_level == 'mid risk':
        recommendations = [
            "Schedule weekly check-ups",
            "Monitor blood pressure daily",
            "Moderate physical activity",
            "Balanced diet recommended"
        ]
    else:
        recommendations = [
            "Regular prenatal check-ups",
            "Maintain healthy lifestyle",
            "Stay hydrated",
            "Continue moderate exercise"
        ]
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
