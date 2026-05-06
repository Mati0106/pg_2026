import optuna
import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score

class OptunaOptimizer:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(),
        )

    def optimizeModel(self):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int('n_estimators', 2, 32),
                "verbosity": 0,
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 100.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 100.0, log=True),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
                # minimum child weight, larger the term more conservative the tree.
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-8, 100, log=True
                ),
                "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
                # defines how selective algorithm is.
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "grow_policy": "depthwise",
                "eval_metric": "logloss",
            }

            xgboost_model = xgb.XGBClassifier(params)
            fold = KFold(n_splits=5, shuffle=True, random_state=0)
            score = cross_val_score(xgboost_model, self.X, self.y, cv=fold, scoring="neg_log_loss")
            return score.mean()

        self.study.optimize(objective, n_trials=3)

    def printOptunaReport(self):
        print('Best params: ', self.study.best_params)
        print('Best value: ', self.study.best_value)
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()

    def getBestParams(self):
        return self.study.best_params



def lasso_model(X_train, X_test, y_train, y_test):
    lasso = Lasso()
    lasso_model = lasso.fit(X_train, y_train)
    lasso_score = lasso_model.score(X_test, y_test)
    print('Lasso score: ', lasso_score)
    import numpy as np
    features = np.array(feature_names)
    non_important = features[lasso.coef_ == 0]
    print('Non important: ', non_important)
