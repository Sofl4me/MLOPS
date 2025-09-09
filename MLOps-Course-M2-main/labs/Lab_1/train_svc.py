# train_svc.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from featurizer import add_svm_features

# 1) Données
ds = load_breast_cancer(as_frame=True)
df = pd.concat([ds['data'], ds['target']], axis=1)
X, y = df.drop(columns=['target']), df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 2) Pipeline SVM : FE + StandardScaler (très important pour SVM)
preproc = Pipeline([
    ('feat_eng', FunctionTransformer(add_svm_features, validate=False)),
    ('scaler', StandardScaler())
])

svc_pipe = Pipeline([
    ('preprocessing', preproc),
    ('classifier', SVC(probability=True, random_state=42))  # probas activées
])

# 3) Grille demandée
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],  # gamma par défaut = 'scale' ok
}

grid = GridSearchCV(
    svc_pipe, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy'
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best = grid.best_estimator_

# 4) Évaluation simple
y_pred = best.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 5) Sauvegarde
dump(best, 'best_svc_cancer_pipeline.joblib')
print("Saved -> best_svc_cancer_pipeline.joblib")
