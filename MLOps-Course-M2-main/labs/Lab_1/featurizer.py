# featurizer.py
import pandas as pd, numpy as np

def add_tree_friendly_features(X_df: pd.DataFrame) -> pd.DataFrame:
    X = X_df.copy()
    eps = 1e-9
    if 'mean area' in X.columns and 'mean perimeter' in X.columns:
        X['area_perimeter_ratio'] = X['mean area'] / (X['mean perimeter'] + eps)
    if 'mean compactness' in X.columns and 'mean concavity' in X.columns:
        X['compactness_x_concavity'] = X['mean compactness'] * X['mean concavity']
    if 'mean texture' in X.columns and 'mean smoothness' in X.columns:
        X['texture_over_smoothness'] = X['mean texture'] / (X['mean smoothness'] + eps)
    if 'mean symmetry' in X.columns:
        X['symmetry_sq'] = X['mean symmetry'] ** 2
    if 'mean radius' in X.columns and 'mean concave points' in X.columns:
        X['radius_x_concavepts'] = X['mean radius'] * X['mean concave points']
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def add_svm_features(X_df: pd.DataFrame) -> pd.DataFrame:
    X = X_df.copy()
    # Une FE simple et différente : produit radius × texture
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    return X
