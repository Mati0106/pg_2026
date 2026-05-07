"""
Hyperparameter Optimization for Regression with Optuna
SHAP values
Modeling
=======================================================
Optimizes LinearRegression, Ridge, Lasso,

"""

import optuna
import numpy as np
import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_data():
    data = pd.read_csv("Housing.csv")
    pd.set_option('display.max_columns', None)
    return data

# ─────────────────────────────────────────────
# Prepare data
# ─────────────────────────────────────────────
def check_n_prepare_data(data):
    # Checking missing values
    missing = data.isnull().sum()
    missing_percent = (missing / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
    missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
    print(missing_df)
    print()

    X, y, df_reshape_data = reshape_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=47) #,random_state=47

    # Standardizing data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Test  samples : {X_test.shape[0]}")
    print(f"  Features      : {X_train.shape[1]}")
    print(f"  Target range  : [{y.min():.2f}, {y.max():.2f}]")

    return X_train, X_test, y_train, y_test, df_reshape_data

def split_data_without_random_state(data):
    X, y, df_reshape_data = reshape_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def reshape_data(data):
    df_reshape_data = data.replace({'yes': 1.0, 'no': 0.0}, inplace=True)
    df_reshape_data = df_reshape_data.replace({'furnished': 2.0, 'semi-furnished': 1.0, 'unfurnished': 0.0},
                                              inplace=True)

    # encoder = LabelEncoder()
    #
    # data['mainroad'] = encoder.fit_transform(data['mainroad'])
    # data['guestroom'] = encoder.fit_transform(data['guestroom'])
    # data['basement'] = encoder.fit_transform(data['basement'])
    # data['hotwaterheating'] = encoder.fit_transform(data['hotwaterheating'])
    # data['airconditioning'] = encoder.fit_transform(data['airconditioning'])
    # data['prefarea'] = encoder.fit_transform(data['prefarea'])
    # data['furnishingstatus'] = encoder.fit_transform(data['furnishingstatus'])
    # df_reshape_data = data

    y = df_reshape_data.iloc[:, 0]  # .to_numpy().astype('float64') ,'furnishingstatus'
    # X = df_reshape_data[['area','bedrooms','bathrooms', 'stories', 'mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea']].values
    X = df_reshape_data.iloc[:, 1:]  # .to_numpy().astype('float64')
    return X,y, df_reshape_data


# ─────────────────────────────────────────────
# Check correlation with all features
# ─────────────────────────────────────────────
def check_correlation(data):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # compute correlation
    corr_matrix = data.corr(method='pearson')

    # init plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Put heatmap with params
    sns.heatmap(corr_matrix, annot=True, ax=ax)#, alpha=1.0, zorder=2)

    # Format
    ax.tick_params(labelsize=9)
    sns.set_theme(font_scale=0.6)

    # display
    plt.show()


# ─────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────
def objective(trial: optuna.Trial, X_train, y_train) -> float:
    """
    Returns the mean CV R² score (higher = better).
    Optuna will MAXIMIZE this value.
    """

    model_name = trial.suggest_categorical(
        "model", ["LinearRegression", "Ridge", "Lasso"]
    )

    if model_name == "LinearRegression":
        # LinearRegression has no regularization hyperparams;
        # we tune fit_intercept and whether to constrain coefficients to be positive.
        fit_intercept = trial.suggest_categorical("lr_fit_intercept", [True, False])
        positive = trial.suggest_categorical("lr_positive", [True, False])
        model = LinearRegression(fit_intercept=fit_intercept, positive=positive)

    elif model_name == "Ridge":
        alpha = trial.suggest_float("ridge_alpha", 1e-4, 1e4, log=True)
        fit_intercept = trial.suggest_categorical("ridge_fit_intercept", [True, False])
        solver = trial.suggest_categorical(
            "ridge_solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
        )
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

    else: #model_name == "Lasso":
        alpha = trial.suggest_float("lasso_alpha", 1e-4, 1e2, log=True)
        fit_intercept = trial.suggest_categorical("lasso_fit_intercept", [True, False])
        max_iter = trial.suggest_int("lasso_max_iter", 500, 5000, step=500)
        selection = trial.suggest_categorical("lasso_selection", ["cyclic", "random"])
        model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            selection=selection,
        )

    # 5-fold CV using R² (negative MSE also works: scoring="neg_mean_squared_error")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    return scores.mean()


# ─────────────────────────────────────────────
# train models without hiperparameters
# ─────────────────────────────────────────────
def train_models_default_hiper_params(X_train, X_test, y_train, y_test):

    print("="*58)
    print()
    print("Training models without hiper-parameters")
    print()
    print("=" * 58)

    # LinearRegression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_train_pred_lr = model_lr.predict(X_train)
    r2_lr = model_lr.score(X_test, y_test)
    # print("odchylenie standardowe: %.4f" % wynik.std())
    print("  LinearRegression -> r^2: %.4f" % r2_lr)
    print("  LinearRegression -> MAE: ", mean_absolute_error(y_train, y_train_pred_lr))
    # cvs = cross_val_score(model_lr, X_test, y_test, cv=5)
    # print("  LinearRegression -> cvs.r^2: %.4f" % cvs.mean())
    print()

    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    y_train_pred_ridge = ridge_model.predict(X_train)
    r2_ridge = ridge_model.score(X_test, y_test)
    print("  Ridge -> r^2: %.4f" % r2_ridge)
    print("  Ridge -> MAE: ", mean_absolute_error(y_train, y_train_pred_ridge))

    # cvs = cross_val_score(ridge_model, X_test, y_test, cv=5)
    # print("  Ridge -> cvs.r^2: %.4f" % cvs.mean())

    print()

    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    y_train_pred_lasso = lasso_model.predict(X_train)
    r2_lasso = lasso_model.score(X_test, y_test)
    print("  Lasso -> r^2: %.4f" % r2_lasso)
    print("  Lasso -> MAE: ", mean_absolute_error(y_train, y_train_pred_lasso))
    # cvs = cross_val_score(lasso_model, X_test, y_test, cv=5)
    # print("  Lasso -> cvs.r^2: %.4f" % cvs.mean())

    print()


# ─────────────────────────────────────────────
# Run the study
# ─────────────────────────────────────────────
def run_study(X_train, y_train, n_trials: int = 80) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="regression_optimization",
        sampler=optuna.samplers.TPESampler(seed=47),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    print(f"\n🔍 Starting optimization ({n_trials} trials)...\n")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study


# ─────────────────────────────────────────────
# Evaluate the best model
# ─────────────────────────────────────────────
def evaluate_best_model(study: optuna.Study, X_train, X_test, y_train, y_test):
    best_params = study.best_params.copy()
    model_name = best_params.pop("model")


    # CV (Cross-Validation – Walidacja krzyżowa)
    # (R²) (Współczynnik determinacji)
    print("\n" + "=" * 58)
    print("  OPTIMIZATION RESULTS")
    print("=" * 58)
    print(f"  Best model    : {model_name}")
    print(f"  Best CV R²    : {study.best_value:.4f}")
    print(f"  Best params   :")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print("=" * 58)

    # ── Strip model-specific prefixes ──────────────────────────
    prefix_strip = {
        "LinearRegression": "lr_",
        "Ridge": "ridge_",
        "Lasso": "lasso_",
    }
    prefix = prefix_strip[model_name]
    clean = {
        k.replace(prefix, ""): v
        for k, v in best_params.items()
        if k.startswith(prefix)
    }

    # ── Build & fit best model ─────────────────────────────────
    model_cls = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,

    }[model_name]

    # BayesianRidge uses max_iter instead of n_iter
    # if model_name == "BayesianRidge" and "n_iter" in clean:
    #     clean["max_iter"] = clean.pop("n_iter")

    best_model = model_cls(**clean)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n  Test Set Metrics:")
    print(f"    R²   : {r2:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print("=" * 58)

    # best_model_without_random_state(best_model)

    return best_model


def best_model_without_random_state(best_model):
    # check best_model without random_state-and-cross_val_score
    X_train_wrs, X_test_wrs, y_train_wrs, y_test_wrs = split_data_without_random_state(load_data())

    # best_model.fit(X_train_wrs, y_train_wrs)
    y_pred_wrs = best_model.predict(X_test_wrs)

    rmse_wrs = np.sqrt(mean_squared_error(y_test_wrs, y_pred_wrs))
    mae_wrs = mean_absolute_error(y_test_wrs, y_pred_wrs)
    r2_wrs = r2_score(y_test_wrs, y_pred_wrs)
    cvs = cross_val_score(best_model, X_test_wrs, y_test_wrs, cv=5)
    # print("  LinearRegression -> cvs.r^2: %.4f" % cvs.mean())
    print("\n")
    print("="*58)
    print("\n  Test model with best Metrics without random_state:")
    # print(f"    R²   : {r2_wrs:.4f}")
    print(f"    cross_val_score R²  : {cvs.mean():.4f}")
    print(f"    RMSE : {rmse_wrs:.4f}")
    print(f"    MAE  : {mae_wrs:.4f}")
    print("=" * 58)
    print("\n\n")

# ─────────────────────────────────────────────
# Top trials leaderboard
# ─────────────────────────────────────────────
def print_top_trials(study: optuna.Study, top_n: int = 5):
    print(f"\n  Top {top_n} Trials:")
    print("  " + "-" * 52)
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )
    for rank, trial in enumerate(sorted_trials[:top_n], start=1):
        mdl = trial.params.get("model", "?")
        print(
            f"  #{rank}  Trial {trial.number:>3d} | {mdl:<15} | R²: {trial.value:.4f}"
        )


# ─────────────────────────────────────────────
# Per-model breakdown
# ─────────────────────────────────────────────
def print_model_breakdown(study: optuna.Study):
    print("\n  Best R² per model type:")
    print("  " + "-" * 40)
    models = ["LinearRegression", "Ridge", "Lasso"]
    for mdl in models:
        trials = [
            t for t in study.trials
            if t.params.get("model") == mdl and t.value is not None
        ]

        if trials:
            best = max(trials, key=lambda t: t.value)
            print(f"  {mdl:<15} → best R²: {best.value:.4f}  (trial #{best.number})")
        else:
            print(f"  {mdl:<15} → no completed trials")


# ─────────────────────────────────────────────
# SHAP-values
# ─────────────────────────────────────────────
def shap_explainer(model,X_train, X_test, y_train, y_test):
    import shap
    import matplotlib.pyplot as plt

    feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                     'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # 4. SHAP Summary Plot (global feature importance)
    plt.figure()  # Create a new figure
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.show()  # Display the plot
    #
    # 5. SHAP Dependence Plot (feature vs. SHAP value)
    shap.dependence_plot('area', shap_values.values, X_test, feature_names=feature_names)
    plt.show()  # Display the plot
    #
    # 6. SHAP Force Plot (local explanation of a single prediction)
    # Create a new figure
    shap.force_plot(explainer.expected_value, shap_values[0].values, X_test[0], feature_names=feature_names,
                    matplotlib=True)
    plt.show()  # Display the plot
    #
    # # 7. SHAP Waterfall Plot (breakdown of individual prediction)
    plt.figure()  # Create a new figure
    shap.plots.waterfall(shap_values[0])
    plt.show()  # Display the plot


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Optuna Regression Hyperparameter Optimization")
    print("  Dataset: California Housing")
    print("  Models : LinearRegression | Ridge | Lasso")
    print("=" * 58)
    print()

    # 1. Data collection
    data = load_data()

    # 2. Feature engineering + Engineering analysis
    # 2.1 #Checking missing values
    # 2.2 #Skalowanie (Scaling): StandardScaler

    # df_reshape_data contains replaced strings in a columns like **yes, no, furnished, semi-furnished, unfurnished** into float values
    X_train, X_test, y_train, y_test, df_reshape_data = check_n_prepare_data(data)

    # 2.3 #heatmap is Feature Selection (FS is a part of Exploratory Data Analysis (EDA))
    check_correlation(df_reshape_data)

    # 3. Modeling
    train_models_default_hiper_params(X_train, X_test, y_train, y_test)

    # 4. Hiperparametry optymalizacyjne
    study = run_study(X_train, y_train, n_trials=120)
    best_model = evaluate_best_model(study, X_train, X_test, y_train, y_test)

    # 5. Interpretacja wyników
    shap_explainer(best_model, X_train, X_test, y_train, y_test)

    # print_top_trials(study, top_n=5)
    # print_model_breakdown(study)

    return study, best_model


if __name__ == "__main__":
    study, best_model = main()

