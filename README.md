# Data-Mining---Python
Repo for various data mining projects utilizing Python. Collaborative. 

# Decision Tree (For future use, ignore)
features = ["Sex", "Family", "IsAlone", "Fare", "Embarked", "Age", "Pclass"]

x = data[features]
y = data["Survived"]

dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree = dtree.fit(x, y)

plt.figure(figsize=(20, 10))
tree.plot_tree(
    dtree, feature_names=features, class_names=["Died", "Survived"], filled=True, rounded=True, fontsize=10,
    impurity=False, proportion=True, label="none"
) 

plt.show()
