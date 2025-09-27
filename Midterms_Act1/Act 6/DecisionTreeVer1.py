import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 

# Function to import the dataset
def importdata():
    url = ""
    data = pd.read_csv(url)

    # Displaying dataset information
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)
    print("First 5 Rows: \n", data.head())
    
    return data

# Consider adding a function to get a sample of churn at 1:.002 

# Data preprocessing
def splitdataset(data):
    # Drop customerID and gender (not useful for prediction)
    data = data.drop("customerID", axis=1)
    data = data.drop("gender", axis=1)

    # Convert columns like Churn, Partner, Dependents, etc. into binary (Yes=1, No=0)
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    data["Partner"] = data["Partner"].map({"Yes": 1, "No": 0})
    data["Dependents"] = data["Dependents"].map({"Yes": 1, "No": 0})
    data["PhoneService"] = data["PhoneService"].map({"Yes": 1, "No": 0})
    data["PaperlessBilling"] = data["PaperlessBilling"].map({"Yes": 1, "No": 0})

    # Convert categorical variables into dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)

    # Features and target
    X = data.drop("Churn", axis=1)
    Y = data["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100, stratify=Y
    )

    return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, y_train):
    # Creating the classifier object (Experiment w/ vars for desired result)
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5
    )

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=5, min_samples_leaf=5
    )

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:", y_pred[:10])
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred)*100)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to plot the decision tree
def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf_object,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        fontsize=8
    )
    plt.show()

if __name__ == "__main__":
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    # Train models
    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = train_using_entropy(X_train, y_train)

    # Predictions
    print("\nGini Model:")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("\nEntropy Model:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

    # Visualizing the Decision Trees
    plot_decision_tree(clf_gini, X.columns, ["No Churn", "Churn"])
    plot_decision_tree(clf_entropy, X.columns, ["No Churn", "Churn"])