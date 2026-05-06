import logging as log

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def lasso_model(X_train, X_test, y_train, y_test, feature_names):
    lasso = Lasso()
    lasso_model = lasso.fit(X_train, y_train)
    lasso_score = lasso_model.score(X_test, y_test)
    log.debug('Lasso score: ', lasso_score)

    features = np.array(feature_names)
    non_important = features[lasso.coef_ == 0]
    log.debug('Non important: ', non_important)
    return lasso_model, lasso_score


def svc_model(X_train, X_test, y_train, y_test, feature_names):
    # Define the parameter grid
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

    # Initialize a SVM classifier
    svm_classifier = SVC()

    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train.values.ravel())

    # Print the best hyperparameters
    log.debug("SVC best Parameters:", grid_search.best_params_)

    # Evaluate the model with the best hyperparameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test.values.ravel(), y_pred)
    log.debug('SVC accuracy: ', accuracy * 100, '%')
    return best_model, accuracy


def xgbclassifier_model(X_train, X_test, y_train, y_test, feature_names):
    params = {
        "n_estimators": 2,
        "max_depth": 3,
    }
    xgboost_model = xgb.XGBClassifier(params)
    xgboost_model.fit(X_train, y_train)

    # 6. Results interpretation (shap values, feature importance, partial dependence plot itp.)
    y_pred = xgboost_model.predict(X_test)
    log.debug('y_pred:', y_pred)

    print(f'Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    ##[
    #    0   1  2      < This happens
    # 0 [41  4  2]
    # 1 [17  6  3]
    # 2 [ 3  3 12]
    # ^ This is what the model predicted
    # ]

    xgb_score = accuracy_score(y_test, y_pred)
    log.info(f'Accuracy: ', xgb_score * 100, '%')

    print('Classification report\n\n', classification_report(y_test, y_pred))
    return xgboost_model, xgb_score
