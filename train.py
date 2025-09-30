# train.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from model_utils import save_model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if 'target' in df.columns:
            y = df['target']
            X = df.drop(columns=['target'])
        elif 'default' in df.columns:
            y = df['default']
            X = df.drop(columns=['default'])
        else:
            raise ValueError("CSV must contain a 'target' or 'default' column.")
        return X, y
    X, y = make_classification(n_samples=5000, n_features=12, n_informative=8,
                               n_redundant=2, n_classes=2, weights=[0.7, 0.3],
                               random_state=42)
    col_names = [f"feat_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=col_names)
    return X, pd.Series(y, name='target')

def build_pipeline(numeric_features):
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
    clf = Pipeline(steps=[('preproc', preprocessor),
                          ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    return clf

def main():
    X, y = load_data(path="data/credit_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    pipeline = build_pipeline(numeric_features)
    pipeline.fit(X_train[numeric_features], y_train)
    preds = pipeline.predict(X_test[numeric_features])
    proba = pipeline.predict_proba(X_test[numeric_features])[:,1]
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    save_model(pipeline, os.path.join(MODEL_DIR, "credit_scoring_model.joblib"))
    print("Model saved to", MODEL_DIR)

if __name__ == "__main__":
    main()
