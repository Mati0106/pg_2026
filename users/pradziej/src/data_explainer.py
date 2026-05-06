import logging as log

import matplotlib.pyplot as plt
import shap


class ShapDataExplainer:
    def __init__(self, model, X_train, X_test, feature_names, target_names, class_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.target_names = target_names
        self.class_names = class_names
        self.shap_values = None

    def shap_explainer(self):
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_test)
        shap_values.feature_names = self.feature_names
        shap_values.target_names = self.target_names
        log.debug('SHAP values shape:', shap_values.shape)
        # shap_values is an instance of Explanation object (https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html#)
        # It's an array of nested arrays. in our case it's:
        # [91 - for each row a shap value is calculated]
        # [*][6] - we have 6 features, so 6 columns
        # [*][*][3] - we have 3 unique values of risk low, mid, high
        self.shap_values = shap_values

    def draw_summary_graphs(self):
        if (self.shap_values is None):
            log.error('Shap values not loaded')
            return;

        # SHAP Summary Plot (global feature importance)
        plt.figure()
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=self.feature_names,
            class_names=self.class_names
        )
        plt.show()

        # SHAP Summary Plot for High Risk feature importance
        plt.figure()
        shap.plots.bar(self.shap_values[:, :, 2])
        plt.show()

        # SHAP High Risk Beeswarm diagram for each feature
        plt.figure()
        shap.plots.beeswarm(self.shap_values[:, :, 2])
        plt.show()
