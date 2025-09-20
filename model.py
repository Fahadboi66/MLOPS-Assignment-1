# train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main(output_path="model.pkl"):
    data = load_iris()
    X, y = data.data, data.target
    # Simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[train.py] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    joblib.dump(model, output_path)
    print(f"[train.py] Saved model to {output_path}")

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    main(output_path="artifacts/model.pkl")
