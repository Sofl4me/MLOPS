from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

def add_combined_feature(X):
    X = X.copy()  # Ensure we're modifying a copy of the DataFrame
    
    # Example feature: combining two features
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    
    return X

# Load the trained model pipeline
model_pipeline = joblib.load('best_cancer_model_pipeline.joblib')

# Define the feature names as per the training data
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
    'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
    'worst fractal dimension'
]


# Define a route for the home page
@app.route('/')
def home():
    return "Welcome to the Breast Cancer Prediction API!"

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    
    # Extract features from the input data
    features = data['features']
    
    # Convert the list of features into a DataFrame
    input_data = pd.DataFrame([features], columns=feature_names)
    
    # Make predictions using the loaded model
    prediction = model_pipeline.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)