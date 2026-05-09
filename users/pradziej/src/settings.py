_dataset_path = "csafrit2/maternal-health-risk-data"

settings = {
    "dataset": _dataset_path,
    "test_size": 0.2,
    "random_state": 42,
    "optuna": {
        "n_trails": 5,
        "min_learning_rate": 0.1,
        "max_learning_rate": 0.3,
        "min_max_depth": 3,
        "max_max_depth": 12,
    }

}
