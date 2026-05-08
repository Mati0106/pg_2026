"""Model training, evaluation, hyperparameter tuning, and business impact simulation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

import src.visualization as visualization
import src.utils as utils
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial logs, show only final result


def split_data(df):
    """Split data into training and test sets (ONLY ONCE!)."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # stratify=y ensures class proportions are preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Fit StandardScaler on training data, apply to both sets. Returns (X_train_scaled, X_test_scaled)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train 4 different ML models and compare their results in a table.

    Logistic Regression is scaled internally (gradient-based, sensitive to feature scale).
    Tree-based models (Decision Tree, Random Forest, XGBoost) receive raw data.
    """
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Define dictionary with our models
    # max_iter=1000 for Logistic Regression to avoid convergence errors
    # l1_ratio=1.0: pure Lasso (L1) — zeros out weak features, acts as built-in feature selection
    # solver='saga': required to support l1_ratio; the only sklearn solver that handles L1/L2/ElasticNet
    # eval_metric='logloss' for XGBoost: monitors training progress using log loss (measures prediction confidence)
    models = {
        "Logistic Regression": (LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=1000, random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
        "Random Forest": (RandomForestClassifier(random_state=42), False),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'), False),
    }

    results = []
    log_reg_y_pred = None

    for name, (model, use_scaling) in models.items():
        X_tr = X_train_scaled if use_scaling else X_train
        X_te = X_test_scaled if use_scaling else X_test

        # 1. Training phase (training on historical data)
        model.fit(X_tr, y_train)

        # 2. Testing phase (testing on hidden data)
        y_pred = model.predict(X_te)

        # 3. Calculating metrics
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Adding results to the list
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1-Score": round(f1, 4)
        })

        if name == "Logistic Regression":
            log_reg_y_pred = y_pred

    # Convert list to readable DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by the best Recall (catching churners)
    results_df = results_df.sort_values(by="Recall", ascending=False).reset_index(drop=True)
    
    print("\n--- MODEL COMPARISON ---")
    print(results_df.to_string())

    utils.save_to_csv(results_df, "initial_model_results.csv")
    assert log_reg_y_pred is not None
    return log_reg_y_pred



def plot_logistic_regression_confusion_matrix(y_test, y_pred) -> None:
    """Plot confusion matrix for Logistic Regression baseline model."""
    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Logistic Regression", "confusion_matrix_initial_logistic_regression.png")


def plot_tuned_xgboost_confusion_matrix(y_test, y_pred) -> None:
    """Plot confusion matrix for the tuned XGBoost model."""
    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Tuned XGBoost", "confusion_matrix_tuned_xgboost.png")


def tune_hyperparameters(X_train, X_test, y_train, y_test):
    """Use Optuna to find the best hyperparameters, optimizing for maximum Recall (catching churners).

    Logistic Regression is scaled internally. Tree-based models receive raw data.
    Each model runs 50 Optuna trials using TPE (Bayesian) sampler.
    """
    print("\n⏳ Starting hyperparameter tuning (Optuna)... This may take a minute!\n")

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    results = []
    best_xgb_model  = None
    best_xgb_y_pred = None

    # --- Logistic Regression ---
    def objective_lr(trial):
        regularization_strength = trial.suggest_float('C', 0.01, 10.0, log=True)
        l1_ratio                = trial.suggest_categorical('l1_ratio', [0.0, 1.0])  # 0.0 = L2/Ridge, 1.0 = L1/Lasso

        # only Optuna-suggested values in params — statics passed directly to model
        params = {
            'C':        regularization_strength,  # inverse of regularization: low = stronger penalty
            'l1_ratio': l1_ratio,                 # l1_ratio=1.0 (Lasso) zeros out weak features; 0.0 (Ridge) shrinks them
        }
        model = LogisticRegression(**params, solver='saga', class_weight='balanced', max_iter=1000, random_state=42)
        return cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall', n_jobs=-1).mean()

    # --- Decision Tree ---
    def objective_dt(trial):
        max_depth         = trial.suggest_int('max_depth', 2, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50)

        params = {
            'max_depth':         max_depth,          # how deep the tree can grow
            'min_samples_split': min_samples_split,  # minimum samples required to split a node
        }
        model = DecisionTreeClassifier(**params, class_weight='balanced', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='recall', n_jobs=-1).mean()

    # --- Random Forest ---
    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth    = trial.suggest_int('max_depth', 3, 20)

        params = {
            'n_estimators': n_estimators,  # number of trees in the forest
            'max_depth':    max_depth,     # depth of each individual tree
        }
        model = RandomForestClassifier(**params, class_weight='balanced', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='recall', n_jobs=-1).mean()

    # --- XGBoost ---
    def objective_xgb(trial):
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth        = trial.suggest_int('max_depth', 3, 10)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 5)  # equivalent of class_weight: count(No)/count(Yes) ≈ 3

        params = {
            'learning_rate':    learning_rate,     # how fast the model learns (smaller = slower but more precise)
            'max_depth':        max_depth,         # maximum depth of a tree, controls model complexity
            'scale_pos_weight': scale_pos_weight,
        }
        model = XGBClassifier(**params, eval_metric='logloss', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='recall', n_jobs=-1).mean()

    models_to_tune = {
        "Logistic Regression": (objective_lr, True),
        "Decision Tree":       (objective_dt, False),
        "Random Forest":       (objective_rf, False),
        "XGBoost":             (objective_xgb, False),
    }

    for name, (objective, use_scaling) in models_to_tune.items():
        print(f"Tuning model: {name}...")

        X_tr = X_train_scaled if use_scaling else X_train
        X_te = X_test_scaled if use_scaling else X_test

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        print(f"  Best Recall (CV avg): {study.best_value:.4f} | Params: {best_params}")

        # Re-train best model on full training data and evaluate on test set
        if name == "Logistic Regression":
            best_model = LogisticRegression(
                **best_params, solver='saga', class_weight='balanced',
                max_iter=1000, random_state=42
            )
        elif name == "Decision Tree":
            best_model = DecisionTreeClassifier(**best_params, class_weight='balanced', random_state=42)
        elif name == "Random Forest":
            best_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
        else:  # XGBoost
            best_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)

        best_model.fit(X_tr, y_train)

        y_pred = best_model.predict(X_te)

        if name == "XGBoost":
            best_xgb_model  = best_model  # save for returning — used by SHAP
            best_xgb_y_pred = y_pred


        results.append({
            "Tuned Model": name,
            "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
            "Recall":      round(recall_score(y_test, y_pred), 4),
            "Precision":   round(precision_score(y_test, y_pred), 4),
            "F1-Score":    round(f1_score(y_test, y_pred), 4),
            "Best Params": str(best_params),
        })

    results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False).reset_index(drop=True)

    print("\n🏆 --- RESULTS AFTER HYPERPARAMETER TUNING (OPTUNA) --- 🏆")
    pd.set_option('display.max_colwidth', None)
    print(results_df.to_string())

    utils.save_to_csv(results_df, "tuned_model_results.csv")
    return best_xgb_model, best_xgb_y_pred

def print_business_impact_simulation(y_test, y_pred_baseline, y_pred_tuned):
    """
    Computes and displays a financial simulation comparing three retention strategies:

    0. Business baseline — give discount to everyone (no ML)
       The simplest real-world alternative: flag every customer as at-risk.
       Catches all churners (FN=0) but wastes budget on non-churners.
    1. Logistic Regression — first ML model (no hyperparameter tuning)
    2. Tuned XGBoost — optimized model (Optuna, maximizing Recall)

    LTV = 1000 (Customer Lifetime Value — full revenue lost if a churner is missed)
    Discount cost = 100 (cost of sending a retention offer)
    """

    cm_base = confusion_matrix(y_test, y_pred_baseline)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)

    # ravel() flattens the 2x2 confusion matrix into a flat list [TN, FP, FN, TP]
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()

    # Strategy 0: give discount to everyone — no ML needed
    # Every churner is caught (TP = all actual churners), every non-churner gets an unnecessary discount (FP = all non-churners)
    tp_all = int(y_test.sum())
    fp_all = int((y_test == 0).sum())
    fn_all = 0

    LTV = 1000
    DISCOUNT_COST = 100
    PROFIT_FROM_RETENTION = LTV - DISCOUNT_COST  # we saved the customer but gave a discount
    COST_OF_FALSE_POSITIVE = DISCOUNT_COST        # unnecessary discount to someone who would stay
    COST_OF_MISSED_CHURNER = LTV                  # customer left, full LTV lost

    profit_all   = (tp_all * PROFIT_FROM_RETENTION) - (fp_all * COST_OF_FALSE_POSITIVE) - (fn_all * COST_OF_MISSED_CHURNER)
    profit_base  = (tp_b * PROFIT_FROM_RETENTION) - (fp_b * COST_OF_FALSE_POSITIVE) - (fn_b * COST_OF_MISSED_CHURNER)
    profit_tuned = (tp_t * PROFIT_FROM_RETENTION) - (fp_t * COST_OF_FALSE_POSITIVE) - (fn_t * COST_OF_MISSED_CHURNER)

    print("\n" + "=" * 60)
    print("Business simulation (model ROI)")
    print("=" * 60)
    print(f"Customer LTV: {LTV} | Campaign cost (discount): {DISCOUNT_COST}\n")

    print("0. Business baseline — give discount to everyone (no ML):")
    print(f"   - Retained customers (TP): {tp_all} -> Profit: ${tp_all * PROFIT_FROM_RETENTION:,}")
    print(f"   - False alarms (FP):       {fp_all} -> Loss:   ${fp_all * COST_OF_FALSE_POSITIVE:,}")
    print(f"   - Missed churners (FN):    {fn_all} -> Loss:   $0")
    print(f"   - Net profit: ${profit_all:,}\n")

    print("1. Logistic Regression (ML baseline, no tuning):")
    print(f"   - Retained customers (TP): {tp_b} -> Profit: ${tp_b * PROFIT_FROM_RETENTION:,}")
    print(f"   - False alarms (FP):       {fp_b} -> Loss:   ${fp_b * COST_OF_FALSE_POSITIVE:,}")
    print(f"   - Missed churners (FN):    {fn_b} -> Loss:   ${fn_b * COST_OF_MISSED_CHURNER:,}")
    print(f"   - Net profit: ${profit_base:,}\n")

    print("2. Tuned XGBoost (optimized model):")
    print(f"   - Retained customers (TP): {tp_t} -> Profit: ${tp_t * PROFIT_FROM_RETENTION:,}")
    print(f"   - False alarms (FP):       {fp_t} -> Loss:   ${fp_t * COST_OF_FALSE_POSITIVE:,}")
    print(f"   - Missed churners (FN):    {fn_t} -> Loss:   ${fn_t * COST_OF_MISSED_CHURNER:,}")
    print(f"   - Net profit: ${profit_tuned:,}\n")

    print("-" * 60)
    print("Summary:")
    print(f"  XGBoost vs give-everyone-a-discount: +${profit_tuned - profit_all:,}")
    print(f"  XGBoost vs Logistic Regression:      +${profit_tuned - profit_base:,}")
    print("=" * 60 + "\n")
