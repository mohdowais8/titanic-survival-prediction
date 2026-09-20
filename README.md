# Titanic Survival Prediction

A beginner Machine Learning project that predicts whether a Titanic passenger survived, using Logistic Regression in Python.

## Dataset
Titanic dataset from Kaggle (`yasserh/titanic-dataset`), 891 passenger records.

## What I Did
1. **Data cleaning:** filled missing Age values with the median, filled missing Embarked values with the mode, and dropped the Cabin column (687 of 891 values missing)
2. **Feature encoding:** converted Sex to 0/1 and Embarked to dummy columns
3. **Train-test split:** 80% training (712 records) and 20% testing (179 records)
4. **Model training:** Logistic Regression using scikit-learn
5. **Evaluation:** accuracy score and confusion matrix

## Features Used
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## Results
- **Test accuracy:** 81.01%
- **Correct predictions:** 145 out of 179
- **Confusion matrix:**

| | Predicted: Not Survived | Predicted: Survived |
|---|---|---|
| **Actual: Not Survived** | 90 | 15 |
| **Actual: Survived** | 19 | 55 |

## Tools Used
Python, Pandas, scikit-learn, VS Code

## How to Run
1. Install the libraries: `pip install pandas scikit-learn`
2. Download the dataset from Kaggle: https://www.kaggle.com/datasets/yasserh/titanic-dataset
3. In `titanic_model.py`, change the CSV file path to where the dataset is saved on your computer
4. Run: `python titanic_model.py`

## Author
Mohd Owais, BCA (AI & ML) student, Galgotias University
