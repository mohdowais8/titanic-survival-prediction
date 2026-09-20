import pandas as pd

df = pd.read_csv(r"C:\Users\Mohd Owais\.cache\kagglehub\datasets\yasserh\titanic-dataset\versions\1\Titanic-Dataset.csv")

print("Before Cleaning:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])

print("After Cleaning:")
print(df.isnull().sum())

df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

print("After encoding:")
print(df.head())
print(df.dtypes)

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Model banao
model = LogisticRegression(max_iter=1000)

# Model ko training data se sikhao
model.fit(X_train, y_train)

print("Model training complete")

# Test data par andaza lagwao
y_pred = model.predict(X_test)

print("First 10 predictions:", y_pred[:10])
print("First 10 actual     :", y_test.values[:10])

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)