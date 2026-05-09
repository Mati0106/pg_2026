import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from src.settings import settings

class OptunaOptimizer:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.study = optuna.create_study(direction="maximize")
        self.params = settings['optuna']

    def optimizeModel(self):
        def objective(trial):
            params = {
                'objective': 'multi:softmax',
                'n_estimators': trial.suggest_int('n_estimators', 2, 100),
                'learning_rate': trial.suggest_float('learning_rate', self.params['min_learning_rate'], self.params['max_learning_rate'], log=True),
                'max_depth': trial.suggest_int('max_depth', self.params['min_max_depth'], self.params['max_max_depth']),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'verbosity': 0,
                'random_state': 42
            }

            xgboost_model = xgb.XGBClassifier(**params)
            score = cross_val_score(xgboost_model, self.X, self.y, n_jobs=-1, cv=3, scoring='accuracy').mean()
            return score

        self.study.optimize(objective, n_trials=self.params['n_trails'])

    def printOptunaReport(self):
        print('Best params: ', self.study.best_params)
        print('Best value: ', self.study.best_value)
        fig = optuna.visualization.plot_optimization_history(self.study, target_name="Accuracy [%]")
        fig.show()

    def getBestParams(self):
        return self.study.best_params
