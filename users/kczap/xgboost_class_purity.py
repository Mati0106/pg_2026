from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from users.kczap.profiling_data.load_honey_dataset import load_honey_data

dataset = load_honey_data()
df_encoded = pd.get_dummies(dataset, columns=["Pollen_analysis"], drop_first=True)

#Convert purity into 3 classes
def purity_to_class(x):
    if x < 0.7:
        return 0
    elif x < 0.9:
        return 1
    else:
        return 2

df_encoded["Purity_class"] = df_encoded["Purity"].apply(purity_to_class)

X = df_encoded.drop(["Purity", "Purity_class", "Price"], axis=1)
Y = df_encoded["Purity_class"]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Initialize the XGBoost classifier
model = XGBClassifier(
    max_depth=5,
    gamma=10
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Classification metrics
print(classification_report(y_test, y_pred))
