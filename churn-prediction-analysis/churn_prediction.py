import pandas as pd

# Load dataset
df = pd.read_csv("data/churn.csv")

# Basic checks
print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())
print("\nChurn value counts:")
print(df['Churn'].value_counts())

print("\nChurn percentage:")
print(df['Churn'].value_counts(normalize=True) * 100)
# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\nEncoded Churn values:")
print(df['Churn'].value_counts())
# Drop customerID (identifier, not a feature)
df.drop('customerID', axis=1, inplace=True)

print("\nColumns after dropping customerID:")
print(df.columns)
# One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nShape after encoding:")
print(df_encoded.shape)
from sklearn.model_selection import train_test_split

# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training completed")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
import joblib

joblib.dump(model, "churn_model.pkl")
print("Model saved successfully")
