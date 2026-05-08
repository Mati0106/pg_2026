# 📚 Project Glossary: Machine Learning Terms

This document contains simple, easy-to-understand definitions of the key concepts used in our Telco Customer Churn project.

## 1. Data Preprocessing & Engineering
* **Data Preprocessing:** Cleaning and preparing raw data so the computer can actually understand and use it.
* **Feature Selection:** Choosing the most important columns (features) for our model and throwing away the useless ones (the noise).
* **Encoding:** Translating text data (like "Yes" or "No") into numbers, because math algorithms can only calculate numbers.
* **Ordinal Encoding:** Changing text to numbers when there is a logical order (e.g., "Month-to-month" = 0, "One year" = 1, "Two year" = 2).
* **One-Hot Encoding:** Creating a new True/False (1 or 0) column for each category when there is no logical order (e.g., turning a "Color" column into "Is_Red" and "Is_Blue").

## 2. Exploratory Data Analysis (EDA)
* **EDA (Exploratory Data Analysis):** The first step after loading data — visually and statistically exploring it to understand its structure, spot patterns, and find relationships between columns before building any model.

## 3. Statistical & Information Theory Concepts
* **Correlation (Pearson, Spearman, Kendall):** Checking if two columns move together (e.g., if one goes up, does the other go up?). Pearson looks for straight lines, while Spearman and Kendall look for general trends.
* **Entropy:** A measure of chaos or unpredictability in the data. It is measured on a scale from **0 to 1** (for binary classification like Yes/No): **0** = perfectly predictable (e.g. everyone churns), **1** = maximum chaos (e.g. exactly 50% churn, 50% stay — hardest to predict). In our dataset, churn is 27% Yes / 73% No, so entropy ≈ 0.84 — fairly high, meaning it's not trivial to predict. Entropy is the foundation of Mutual Information and Decision Trees — they split data to reduce entropy as much as possible.
* **Mutual Information:** A smart score that tells us: *"If I know the value of column A, how much does it help me guess column B?"*. It easily catches hidden patterns that normal correlation misses.
* **Stratification (Stratified Split):** Making sure that when we split our data into training and testing parts, both parts have the exact same percentage of churned customers. It prevents the model from learning on bad proportions.

## 4. Machine Learning Algorithms
* **Binary Classification:** A type of ML problem where the model must assign each example to one of exactly **two** classes — in our case: *churn (1)* or *stay (0)*. The model outputs a probability (e.g. 0.82) and a decision threshold (default 0.5) converts it to a label. Contrasted with *multiclass* (3+ categories) or *regression* (predicting a number).
* **Baseline Model:** A basic, simple model used as a starting point. We use it to see if more complex models are actually doing a better job or just overcomplicating things.
* **Logistic Regression:** A simple math model that predicts the probability of a Yes/No answer (like *"Will the customer leave?"*). Internally, it assigns a **weight** to each feature (e.g., *monthly charges* gets a high weight, *gender* gets a low one), multiplies them together, and feeds the result into a **sigmoid function** — an S-shaped curve that squashes any number into a value between 0 and 1. That value is the predicted probability: e.g. 0.82 means *"82% chance this customer churns"*. If the probability is above 0.5, the model predicts "Yes" (churn); otherwise "No" (stays). The weights are learned automatically during training by minimizing prediction errors.
* **Decision Tree:** A model that makes decisions by asking a series of Yes/No questions, looking very much like a simple flowchart.
* **Random Forest:** A "forest" made of many different Decision Trees. They all vote on the final answer, which makes the prediction much more accurate and stable.
* **XGBoost:** A very powerful, advanced algorithm where multiple trees are built one after another, and each new tree tries to fix the mistakes made by the previous one.

## 5. Regularization
* **Regularization:** A technique that penalizes overly complex models to prevent overfitting — the model is forced to keep its weights small and focus only on the most important features.
* **Lasso (L1 Regularization):** A **type of regularization** — not a hyperparameter. It defines *how* the model is penalized: unimportant features get their weights pushed to exactly **zero**, removing them from the model automatically. The penalty term added to the loss function is the **absolute value** of each weight (`λ × Σ|wᵢ|`). The optimizer finds it cheaper to zero out weak weights than to keep paying a constant cost for them. Controlled by parameter `C` (inverse of penalty strength): small `C` = strong penalty = more features zeroed out. The hyperparameter that *selects* whether to use Lasso is `l1_ratio` — Lasso itself is just the regularization strategy.
* **Ridge (L2 Regularization):** Similar to Lasso, but the penalty is based on the **square** of each weight (`λ × Σwᵢ²`). Instead of zeroing features out, it just shrinks all weights proportionally. Features are never fully removed — just reduced.
* **ElasticNet:** A mix of L1 and L2 regularization, controlled by `l1_ratio` (0.0 = pure Ridge, 1.0 = pure Lasso, 0.5 = 50/50 mix). In our Logistic Regression we use `l1_ratio=1.0` — pure Lasso behavior via the ElasticNet interface. Requires `solver='saga'` — the only sklearn solver that supports ElasticNet.
* **`l1_ratio`:** Parameter in ElasticNet controlling the mix between L1 and L2. Valid range: 0.0 to 1.0. Value of `1.0` means pure Lasso (zeros out weak features); `0.0` means pure Ridge (shrinks all weights). During hyperparameter tuning, Optuna chooses between `[0.0, 1.0]` to find the better regularization type automatically.
* **`solver='saga'`:** The optimization algorithm used internally by Logistic Regression to find the best weights. `saga` (Stochastic Average Gradient Augmented) is the only sklearn solver that supports `l1_ratio` / ElasticNet — required whenever `l1_ratio` is used.

## 6. Hyperparameter Tuning
* **Hyperparameters:** Settings that control how a model learns — they are set before training and are not learned from data (e.g. `max_depth` for Decision Tree, `learning_rate` for XGBoost).
* **Overfitting:** When a model memorizes the training data too well — including random noise — and then performs poorly on new, unseen data. Like a student who memorizes answers instead of understanding the subject.
* **Cross-Validation (CV):** A technique for evaluating model quality without touching the test set. The training data is split into `k` equal parts (folds). The model is trained `k` times — each time on `k-1` folds and tested on the remaining 1. The final score is the average across all folds. We use `cv=5` — 5 rounds, each using a different 20% as validation.
* **GridSearchCV:** Exhaustive search — tries every possible combination of hyperparameters from a predefined list. Slow for large grids but guaranteed to find the best combination within the list.
* **Optuna:** Intelligent hyperparameter tuning library using **Bayesian optimization (TPE sampler)**. Instead of trying all combinations, it learns from previous trials which parameter regions are promising and focuses search there. Faster than GridSearch and can search continuous ranges (e.g. `C` from 0.01 to 10.0 instead of a fixed list).
* **TPE (Tree-structured Parzen Estimator):** The default Bayesian sampler used by Optuna. After an initial random exploration phase (~10 trials), TPE builds two probability models — one for "good" results and one for "bad" — and uses them to select the next parameters most likely to improve the score. This is why it finds good results much faster than random or grid search.
* **`n_trials`:** The number of Optuna trials (attempts) per model. Each trial = one set of hyperparameters → train model → evaluate Recall via cross-validation. We use `n_trials=50` per model (200 total across 4 models). Sufficient for 2–3 hyperparameters; more complex search spaces would require 200–500.

### Hyperparameters used in this project

* **`C` (Logistic Regression):** Controls how much the model is allowed to "memorize" training data. Think of it as how strict the teacher is:
  * **Low C (e.g. 0.01)** = strict teacher = model must be simple, ignores details, less risk of overfitting
  * **High C (e.g. 10.0)** = lenient teacher = model can be more complex, may overfit
  * Our best value: **C=0.604** — slightly strict, simple model but not oversimplified
  * Note: C is the *inverse* of penalty strength — `penalty = 1/C`

* **`max_depth` (Decision Tree / Random Forest / XGBoost):** How many levels of questions the tree can ask before making a decision. A shallow tree (e.g. depth=2) asks only 2 questions — simple, less likely to overfit. A deep tree asks many questions — can memorize training data.
  * Our best: Decision Tree `max_depth=2`, Random Forest `max_depth=3`, XGBoost `max_depth=4`

* **`min_samples_split` (Decision Tree):** Minimum number of customers required before the tree is allowed to split a node into two branches. High value = fewer, broader splits = simpler model.
  * Our best: **min_samples_split=8** — a node needs at least 8 customers to be split further

* **`n_estimators` (Random Forest):** How many decision trees are in the forest. More trees = more stable "majority vote" = better results, but slower training.
  * Our best: **n_estimators=78**

* **`learning_rate` (XGBoost):** How big of a step the model takes when learning from each mistake. Small learning rate = slow but precise learning. Large learning rate = fast but may overshoot the optimal solution.
  * Our best: **learning_rate=0.012** — very slow and careful learning

* **`scale_pos_weight` (XGBoost):** Tells XGBoost how much more important churners are compared to non-churners. Value = count(non-churners) / count(churners) ≈ 2.77. We let Optuna tune it in range 1–5, best found: **5** (slightly above natural ratio, pushing model harder toward catching churners).

## 7. Model Explainability
* **SHAP (SHapley Additive exPlanations):** A method that explains individual model predictions by calculating how much each feature contributed to pushing the prediction above or below the average. Based on **Shapley values** from game theory.
* **Shapley Values:** For each prediction, SHAP assigns a score to every feature: positive score = this feature pushed the model toward predicting churn; negative score = pushed toward staying. The sum of all SHAP values equals the difference between the prediction and the average prediction across all customers.
* **SHAP Summary Plot:** A global view — shows which features matter most across all predictions in the test set. Each dot is one customer; color shows the feature value (red = high, blue = low); position on X-axis shows SHAP impact.
* **SHAP Dependence Plot:** Shows how one specific feature (e.g. `tenure`) affects predictions across its full range of values — reveals non-linear patterns (e.g. churn drops sharply after 12 months of tenure).
* **SHAP Force Plot:** A local explanation for one specific customer — shows which features pushed the prediction up (toward churn) and which pushed it down (toward staying), and by how much.
* **SHAP Waterfall Plot:** Same as Force Plot but displayed as a vertical waterfall chart — each bar shows the contribution of one feature, starting from the average prediction and ending at the final prediction for this customer.

## 8. Evaluation Metrics
* **Confusion Matrix:** A 2×2 table that shows exactly where the model is right and where it makes mistakes. Rows = what actually happened, Columns = what the model predicted. Four cells: TN (correctly predicted stays), FP (false alarms), FN (missed churners), TP (correctly predicted churns). It is the foundation for all other metrics — Accuracy, Recall, Precision, and F1 are all calculated from these 4 numbers.
* **Accuracy:** Overall, out of all the model's guesses, what percentage was exactly right? E.g. 1000 customers total: 700 stayed → model correctly predicted 650 of them; 300 churned → model correctly predicted 150 of them. Total correct: 650 + 150 = 800. Accuracy = 800 / 1000 = **80%**.
* **Recall (Sensitivity):** Out of all the customers who *actually* left, how many did our model successfully catch? (This is the most important score for predicting churn). E.g. 1000 customers actually churned → model correctly flagged 800 of them. Recall = 800 / 1000 = **80%**. The remaining 200 were missed (FN) — the most costly mistake.
* **Precision:** When our model yells *"This customer will leave!"*, how often is it actually right? E.g. 1000 customers actually churned, model flagged 800 as "will churn" → all 800 actually churned (zero false alarms). Precision = 800 / 800 = **100%** — but Recall is only 80% because 200 churners were missed. Another example: model flagged 900, but 100 of them stayed → Precision = 800 / 900 = **89%**. Low precision means wasting money on discounts for customers who weren't leaving anyway.
* **F1-Score:** A combined score that balances Recall and Precision into one number. Useful when both missing churners (low Recall) and false alarms (low Precision) are costly. Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`. E.g. Precision = 90%, Recall = 80% → F1 = 2 × (0.9 × 0.8) / (0.9 + 0.8) = **84.7%**. A model with Recall = 100% but Precision = 10% would have F1 = only 18% — it catches everyone but raises too many false alarms to be useful.
* **TP (True Positive):** Model predicted churn → customer actually churned ✅
* **TN (True Negative):** Model predicted stay → customer actually stayed ✅
* **FP (False Positive):** Model predicted churn → customer actually stayed ❌ (false alarm — we sent a discount unnecessarily)
* **FN (False Negative):** Model predicted stay → customer actually churned ❌ (missed churner — the most costly mistake)

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Recall    = TP / (TP + FN)
Precision = TP / (TP + FP)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```