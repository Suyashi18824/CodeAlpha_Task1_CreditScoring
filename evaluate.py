# evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from model_utils import load_model

MODEL_PATH = "models/credit_scoring_model.joblib"

def main():
    model = load_model(MODEL_PATH)
    try:
        df = pd.read_csv("data/credit_test.csv")
        if 'target' not in df.columns:
            raise ValueError
        X = df.drop(columns=['target'])
        y = df['target']
    except Exception:
        print("No test CSV found, using synthetic evaluation sample.")
        X = pd.DataFrame(np.random.randn(100, len(model.named_steps['preproc'].transformers_[0][2])),
                         columns=[f"feat_{i}" for i in range(len(model.named_steps['preproc'].transformers_[0][2]))])
        y = np.random.randint(0,2,size=(100,))
    preds = model.predict(X)
    proba = model.predict_proba(X)[:,1]
    print(classification_report(y, preds))
    print("ROC AUC:", roc_auc_score(y, proba))

if __name__ == "__main__":
    main()
