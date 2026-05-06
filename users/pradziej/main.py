import logging as log

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_converter import DataConverter
from src.data_explainer import ShapDataExplainer
from src.data_loader import DataLoader
from src.models import lasso_model
from src.models import svc_model
from src.models import xgbclassifier_model
from src.settings import settings

log.basicConfig(level=log.INFO)


def run():
    # 1. Read the data
    data_loader = DataLoader()
    df: pd.DataFrame = data_loader.load_data(settings['dataset'], force_download=False)
    data_loader.print_report()
    data_loader.clean_data()

    # 2. Data analysis + preprocessing
    data_converter = DataConverter()
    # Convert a subset of columns to categorical
    RiskLevelConversionMap = {
        'low risk': 0,
        'mid risk': 1,
        'high risk': 2
    }
    data_converter.encode_categorial_with_map(df, 'RiskLevel', RiskLevelConversionMap)

    # 3. Feature engineering
    target_names = ['RiskLevel']
    feature_names = [c for c in df.columns if c not in target_names]
    print(f'Feature names: {feature_names}, Target names: {target_names}')
    [print(f'RiskLevel correlation with {x}\n', df[['RiskLevel', x]].corr(method='spearman')) for x in feature_names]

    X, y = df[feature_names], df[target_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=settings['test_size'],
                                                        random_state=settings['random_state']
                                                        )

    # Shape - shows amount of rows and columns (works like list length, but for matrix)
    print(f'Training Shape X:', X_train.shape, 'Testing Shape X:', X_test.shape)
    print(f'Training Shape y:', y_train.shape, 'Testing Shape y:', y_test.shape)

    # StandardScaler
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # 4. Modeling
    xgboost_model, xgb_score = xgbclassifier_model(X_train, X_test, y_train, y_test, feature_names)
    _svc_model, svc_score = svc_model(X_train, X_test, y_train, y_test, feature_names)
    _lasso_model, _lasso_score = lasso_model(X_train, X_test, y_train, y_test, feature_names)

    plt.figure()
    plt.bar(["SVC", "XGBClassifier"], [svc_score, xgb_score])
    plt.show()

    # 5. Optimization (with Optuna)
    # optuna_optimizer = OptunaOptimizer(X,y)
    # optuna_optimizer.optimizeModel()
    # optuna_optimizer.printOptunaReport()
    # xgb_params = optuna_optimizer.getBestParams()

    # 6.3. Explain the model's predictions using SHAP
    data_explainer = ShapDataExplainer(xgboost_model, X_train, X_test, feature_names, target_names,
                                       list(RiskLevelConversionMap.keys()))
    data_explainer.shap_explainer()
    data_explainer.draw_summary_graphs()


if __name__ == '__main__':
    run()
