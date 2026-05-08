import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from users.kczap.profiling_data.load_honey_dataset import load_honey_data
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#  DATA PREPARATION
def load_and_prepare_data():
    df = load_honey_data()
    df_encoded = pd.get_dummies(df, columns=["Pollen_analysis"], drop_first=True)
    df_encoded = df_encoded.astype(float)
    X = df_encoded.drop(["Purity", "Price"], axis=1)
    y = df_encoded["Purity"]
    return X, y

def split_data(X, y, seed=7, test_size=0.33):
    return train_test_split(X, y, test_size=test_size, random_state=seed)



#  REGRESSION MODELS
def train_baseline_model(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }



def train_final_model(X_train, y_train, best_params):
    final_model = XGBRegressor(
        objective='reg:squarederror',
        **best_params
    )
    final_model.fit(X_train, y_train)
    return final_model



#  SHAP
def plot_shap_all(model, X_train, X_test, feature_names, feature="Density", index=0):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    # plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Dependence plot
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_test,
        feature_names=feature_names
    )
    plt.show()

    # Force plot
    shap.force_plot(
        explainer.expected_value,
        shap_values[index].values,
        X_test.iloc[index],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    # plt.savefig("shap_force.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    # plt.savefig("shap_waterfall.png", dpi=300, bbox_inches='tight')
    plt.show()

#  CLASSIFICATION MODELS
def train_logistic_model(X_train, y_train):
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


#  CLASSIFICATION DATA PREP
def load_and_prepare_classification_data():
    df = load_honey_data()
    df_encoded = pd.get_dummies(df, columns=["Pollen_analysis"], drop_first=True)
    df_encoded = df_encoded.astype(float)
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
    # Convert purity into 3 classes
    def purity_to_class(x):
        if x < 0.7:
            return 0
        elif x < 0.9:
            return 1
        else:
            return 2

    df_encoded["Purity_class"] = df_encoded["Purity"].apply(purity_to_class)
    X = df_encoded.drop(["Purity", "Purity_class", "Price"], axis=1)
    y = df_encoded["Purity_class"]

    return X, y

