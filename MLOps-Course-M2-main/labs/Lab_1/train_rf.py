# train_rf.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from featurizer import add_tree_friendly_features  # doit être importable

# data
ds = load_breast_cancer(as_frame=True)
df = pd.concat([ds['data'], ds['target']], axis=1)
X, y = df.drop(columns=['target']), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# pipeline: feature eng + RandomForest (pas de scaler pour arbres)
preproc = Pipeline([('feat_eng', FunctionTransformer(add_tree_friendly_features, validate=False))])
pipe = Pipeline([('preprocessing', preproc), ('classifier', RandomForestClassifier(random_state=42))])

# grille demandée
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid.fit(X_train, y_train)

best = grid.best_estimator_
dump(best, 'best_rf_cancer_pipeline.joblib')
print("Saved best model -> best_rf_cancer_pipeline.joblib")