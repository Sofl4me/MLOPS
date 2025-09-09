from flask import Flask, request, jsonify
import joblib
import pandas as pd
#from featurizer import add_tree_friendly_features  # <-- IMPORTANT pour joblib.load
from featurizer import add_svm_features  # <-- IMPORTANT pour joblib.load

app = Flask(__name__)

# Charge le **nouveau** modèle RF
#MODEL_PATH = "best_rf_cancer_pipeline.joblib"
MODEL_PATH = "best_svc_cancer_pipeline.joblib"
model_pipeline = joblib.load(MODEL_PATH)
print("Loaded classifier:", type(model_pipeline.named_steps['classifier']).__name__)


# Les 30 features attendues quand tu envoies "features: [...]"
feature_names = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

@app.route('/')
def home():
    return "Welcome to the Breast Cancer Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    input_df = pd.DataFrame([features], columns=feature_names)
    pred = model_pipeline.predict(input_df)
    return jsonify({'prediction': int(pred[0])})

if __name__ == '__main__':
    # Pour éviter les surprises de reloader/port, je mets debug=False ici
    app.run(host='127.0.0.1', port=5001, debug=False)
