"""
Titanic Survival Prediction
Uses archive.zip containing Titanic-Dataset.csv, or a directly available CSV.
Install dependencies:
    python -m pip install pandas scikit-learn joblib
Run:
    python titanic_survival.py
"""
from pathlib import Path
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "archive.zip"
CSV_PATH = BASE_DIR / "Titanic-Dataset.csv"

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
elif ZIP_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV file was found inside archive.zip")
        with archive.open(csv_names[0]) as csv_file:
            df = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(
        "Place archive.zip or Titanic-Dataset.csv in the same folder as this script."
    )

print(f"Dataset shape: {df.shape}")
print("\nMissing values in selected columns:")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"
print(df[features + [target]].isna().sum())

X = df[features]
y = df[target]

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print(f"\nTraining passengers: {len(X_train)}")
print(f"Testing passengers: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
print("\nConfusion matrix (rows=actual, columns=predicted; order 0, 1):")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["Did not survive", "Survived"]))

model_path = BASE_DIR / "titanic_survival_model.joblib"
joblib.dump(model, model_path)
print(f"\nSaved trained model to: {model_path}")

# Example prediction for a hypothetical passenger (not a real historical record).
example = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
}])
survival_prediction = model.predict(example)[0]
survival_probability = model.predict_proba(example)[0, 1]
print("\nExample hypothetical passenger:")
print("Prediction:", "Survived" if survival_prediction == 1 else "Did not survive")
print(f"Estimated survival probability: {survival_probability:.1%}")
