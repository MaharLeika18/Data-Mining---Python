import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 
import textwrap

# Function to import the dataset
def importdata():
    url = "https://raw.githubusercontent.com/MaharLeika18/Data-Mining---Python/refs/heads/Loue/Midterms_Act1/Act%206/Telco-Customer-Churn.csv"
    data = pd.read_csv(url)

    # Displaying dataset information
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)
    print("First 5 Rows: \n", data.head())
    
    return data

# Data preprocessing
def splitdataset(data):
    # Drop irrelevant columns (not useful for prediction)
    data = data.drop("customerID", axis=1)
    data = data.drop("gender", axis=1)
    data = data.drop("SeniorCitizen", axis=1)

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
        criterion="gini", random_state=100, max_depth=4, min_samples_leaf=10
    )

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=4, min_samples_leaf=10
    )

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def train_with_pruning(X_train, y_train, ccp_alpha=0.01):  
    clf_pruned = DecisionTreeClassifier(
        criterion="gini", random_state=100, ccp_alpha=ccp_alpha  
    )
    clf_pruned.fit(X_train, y_train)
    return clf_pruned

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
def plot_decision_tree(clf_object, feature_names, class_names, max_width=15):
    plt.figure(figsize=(20, 10))
    clean_tree = plot_tree(
        clf_object,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        fontsize=8,
    )

    # Remove gini, samples, value from text
    for tree in clean_tree:
        if hasattr(tree, "get_text"):
            txt = tree.get_text()
            clean_txt = txt.split("\n")[0]
            wrapped_txt = "\n".join(textwrap.wrap(clean_txt, width=max_width))
            tree.set_text(wrapped_txt)

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

    # Visualizing the Decision Trees using auto-tuning pruning
    path = clf_gini.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    best_alpha = None
    best_acc = 0

    for alpha in ccp_alphas:
        clf_temp = DecisionTreeClassifier(random_state=100, ccp_alpha=alpha)
        clf_temp.fit(X_train, y_train)
        acc = clf_temp.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
        print(f"Alpha={alpha:.5f}, Test Accuracy={acc:.3f}")

    print(f"\nBest alpha: {best_alpha:.5f} with Test Accuracy={best_acc:.3f}")

    clf_pruned = train_with_pruning(X_train, y_train, ccp_alpha=best_alpha)
    y_pred_pruned = prediction(X_test, clf_pruned)
    print("\n--- Pruned Model ---")  
    cal_accuracy(y_test, y_pred_pruned)

    plot_decision_tree(clf_gini, X.columns, ["No Churn", "Churn"])
    plot_decision_tree(clf_entropy, X.columns, ["No Churn", "Churn"])
    plot_decision_tree(clf_pruned, X.columns, ["No Churn", "Churn"])