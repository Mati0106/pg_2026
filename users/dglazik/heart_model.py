import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve,
)

import optuna

import shap

optuna.logging.set_verbosity(optuna.logging.WARNING)

CWD = os.getcwd()
DATA_PATH = f"{CWD}/heart.csv"
PLOTS_DIR = f"{CWD}/plots"
RANDOM_STATE = 77

NUM_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]
TARGET = "HeartDisease"

def save_fig(filename):
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.close()

def build_preprocessor():
    # Numeric: median imputation + standardization. Categorical: mode imputation + one-hot encoding.
    # The function prevents data leakage (automaticaly fit scaler and encoder on train set only, transform on test).
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # create separate columns for each category (e.g. Sex_F, Sex_M)
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset loaded from:", path)
    print("Shape:", df.shape)
    return df

def analyze_and_preprocess(df):
    print("\nColumn types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

    #replace all zeroes with Nan - only in "Cholesterol" and "RestingBP"
    for col in ["Cholesterol", "RestingBP"]:
        n = (df[col] == 0).sum()
        if n > 0:
            df[col] = df[col].replace(0, np.nan)
            print(f"\n{col}: {n} zero values replaced with NaN")

    #replace 0/1 in FastingBS, with explicit categorical value
    df["FastingBS"] = df["FastingBS"].replace({0: "OK", 1: "NOK"})
    print("\nFastingBS replaced: 0 -> 'OK', 1 -> 'NOK'")

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[NUM_FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    save_fig("01_correlation_heatmap.png")
    print("Plot saved: 01_correlation_heatmap.png")

    return df[NUM_FEATURES + CAT_FEATURES], df[TARGET]


def evaluate_baseline(X_train, y_train):
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    """Class-wise stratified K-Fold cross-validator.
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class in `y` in a binary or multiclass classification
    setting.
    """    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=["roc_auc", "accuracy", "f1"],
                            return_train_score=False, n_jobs=-1)
    print(f"\nRandomForest Cross-Validation:"
          f"  AUC = {cv_res['test_roc_auc'].mean():.4f} (+/- {cv_res['test_roc_auc'].std():.4f})"
          f"  Accuracy = {cv_res['test_accuracy'].mean():.4f}"
          f"  F1 = {cv_res['test_f1'].mean():.4f}")


def optimize(X_train, y_train):
    print("\nOptimizing...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", classifier)])
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    optimization_process = optuna.create_study(direction="maximize")
    optimization_process.optimize(objective, n_trials=10, timeout=180, show_progress_bar=True)

    print("Best parameters:", optimization_process.best_params)
    print("Best AUC (CV):", round(optimization_process.best_value, 4))

    best_params = optimization_process.best_params
    tuned_clf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    tuned_pipe = Pipeline([("pre", build_preprocessor()), ("clf", tuned_clf)])

    return tuned_pipe


def train_final_model(tuned_pipe, X_train, y_train, X_test):
    print("\nFinal training with optimized hyperparameters...")
    tuned_pipe.fit(X_train, y_train)
    y_pred = tuned_pipe.predict(X_test)
    y_prob = tuned_pipe.predict_proba(X_test)[:, 1]
    return tuned_pipe, y_pred, y_prob


def interpret_results(y_test, y_pred, y_prob):

    print("\nTest set evaluation:")
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("PR AUC:", round(average_precision_score(y_test, y_prob), 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="AUC = " + str(round(roc_auc_score(y_test, y_prob), 4)))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - RandomForest")
    plt.legend()
    save_fig("02_roc_curve.png")
    print("Plot saved: 02_roc_curve.png")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No HD", "HD"], yticklabels=["No HD", "HD"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    save_fig("03_confusion_matrix.png")
    print("Plot saved: 03_confusion_matrix.png")

def compute_shap(tuned_pipe, X_test):
    print("\nComputing SHAP values...")
    pre_step = tuned_pipe.named_steps["pre"]
    feat_names = pre_step.get_feature_names_out()
    X_test_t = pre_step.transform(X_test)
    sample = X_test_t[:200] if len(X_test_t) > 200 else X_test_t

    clf_step = tuned_pipe.named_steps["clf"]
    explainer = shap.TreeExplainer(clf_step)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feat_names, show=False)
    plt.title("SHAP - Feature Impact (Beeswarm)")
    save_fig("04_shap_summary_beeswarm.png")
    print("Plot saved: 04_shap_summary_beeswarm.png")

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).flat[-1])
    exp = shap.Explanation(
        values=shap_values[0], base_values=base_val,
        data=sample[0], feature_names=list(feat_names),
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.title("SHAP - Waterfall (Patient 0)")
    save_fig("05_shap_waterfall_patient.png")
    print("Plot saved: 05_shap_waterfall_patient.png")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    print("\nTop 5 features by SHAP importance:")
    for i in top_idx:
        print("  " + feat_names[i] + ": mean |SHAP| =", round(float(mean_abs[i]), 4))


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. load data
    df = load_data(DATA_PATH)

    # 2. analyze and clean data
    X, y = analyze_and_preprocess(df)

    # 3. prepare training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    print("\nTraining samples:", len(X_train), "| Test samples:", len(X_test))

    # 4. prepare benchmark
    evaluate_baseline(X_train, y_train)

    # 5. optuna - optimize hyperparameters
    tuned_pipe = optimize(X_train, y_train)

    # 6. perform final training, with use of optimized hyperparameters
    tuned_pipe, y_pred, y_prob = train_final_model(tuned_pipe, X_train, y_train, X_test)

    # 7. display results
    interpret_results(y_test, y_pred, y_prob)

    # 8. analyze SHAP
    compute_shap(tuned_pipe, X_test)

    print("\nSummary:")
    print("  Model: RandomForest")
    print("  ROC AUC (test):", round(roc_auc_score(y_test, y_prob), 4))
    print("  PR AUC (test):", round(average_precision_score(y_test, y_prob), 4))
    print("  Plots saved to:", PLOTS_DIR)

    # 9. predict fate of sample patient
    sample_patient = {
        "Age": 49, "Sex": "M", "ChestPainType": "ASY",
        "RestingBP": 130, "Cholesterol": 80, "FastingBS": "NOK",
        "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
        "Oldpeak": 0.0, "ST_Slope": "Down",
    }
    prob = tuned_pipe.predict_proba(pd.DataFrame([sample_patient]))[0, 1]
    print("Demo patient - probability of heart disease:", round(prob, 4))

if __name__ == "__main__":
    main()
