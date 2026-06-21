import os, json
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from huggingface_hub import hf_hub_download, create_repo, upload_file
from src.config import DATASET_REPO_ID, MODEL_REPO_ID, TARGET_COLUMN, RANDOM_STATE

HF_TOKEN = os.getenv("HF_TOKEN")
train_path = hf_hub_download(DATASET_REPO_ID, "train.csv", repo_type="dataset", token=HF_TOKEN) if HF_TOKEN else "data/train.csv"
test_path = hf_hub_download(DATASET_REPO_ID, "test.csv", repo_type="dataset", token=HF_TOKEN) if HF_TOKEN else "data/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN].astype(int)
X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN].astype(int)

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])

model_spaces = {
    "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"), {
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 10]
    }),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1), {
        "model__n_estimators": [50],
        "model__max_depth": [8, None]
    }),
    "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE), {
        "model__n_estimators": [50],
        "model__learning_rate": [0.1, 0.5]
    })
}

os.makedirs("models", exist_ok=True)
os.makedirs("tracking", exist_ok=True)
os.makedirs("reports", exist_ok=True)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
experiment_results = []
best_model = None
best_f1 = -1

for model_name, (estimator, params) in model_spaces.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    grid = GridSearchCV(pipeline, params, cv=cv, scoring="f1", n_jobs=1)
    grid.fit(X_train, y_train)
    preds = grid.predict(X_test)
    probs = grid.predict_proba(X_test)[:, 1]
    metrics = {
        "model_name": model_name,
        "best_cv_f1": grid.best_score_,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "best_params": grid.best_params_
    }
    experiment_results.append(metrics)
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_model = grid.best_estimator_
        best_metrics = metrics
        best_predictions = preds

pd.DataFrame([{k: v for k, v in row.items() if k != "best_params"} for row in experiment_results]).to_csv("tracking/experiment_results.csv", index=False)
with open("tracking/experiment_results.json", "w") as f:
    json.dump(experiment_results, f, indent=2, default=str)
joblib.dump(best_model, "models/model.joblib")
with open("models/model_metrics.json", "w") as f:
    json.dump(best_metrics, f, indent=2, default=str)
with open("reports/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, best_predictions))
pd.DataFrame(confusion_matrix(y_test, best_predictions)).to_csv("reports/confusion_matrix.csv", index=False)

print("Best model:", best_metrics["model_name"])
print("Best metrics:", best_metrics)

if HF_TOKEN:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", token=HF_TOKEN, exist_ok=True, private=False)
upload_file(
    path_or_fileobj="models/model.joblib",
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
upload_file(
    path_or_fileobj="models/model_metrics.json",
    path_in_repo="model_metrics.json",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
upload_file(
    path_or_fileobj="tracking/experiment_results.csv",
    path_in_repo="experiment_results.csv",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
print(f"Model registered: https://huggingface.co/{MODEL_REPO_ID}")
