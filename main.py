#!/usr/bin/python3
"""Modified Decision Tree Classifier for the Adult Census Dataset."""

import pandas as pd
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


adult_data = pd.read_csv('adult.csv')

# Handle missing values (numeric and categorical features separately)

numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_nominal = ['workclass', 'occupation', 'native.country', 'marital.status', 'relationship', 'race', 'sex']
categorical_ordinal = ['education']

median_imputer = SimpleImputer(strategy='median')
adult_data[numerical_features] = median_imputer.fit_transform(adult_data[numerical_features])

imputer_nominal = SimpleImputer(strategy='most_frequent')
adult_data[categorical_nominal] = imputer_nominal.fit_transform(adult_data[categorical_nominal])

imputer_ordinal = SimpleImputer(strategy='most_frequent')
adult_data[categorical_ordinal] = imputer_ordinal.fit_transform(adult_data[categorical_ordinal])

# Encode ordinal categorical features (education)
education_levels = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 
    'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Masters', 
    'Prof-school', 'Doctorate'
]

# Ordinal Encoding to the 'education' feature
ordinal_encoder = OrdinalEncoder(categories=[education_levels])
adult_data['education'] = ordinal_encoder.fit_transform(adult_data[['education']])

# One-Hot Encoding for nominal categorical features
adult_data = pd.get_dummies(adult_data, columns=categorical_nominal, drop_first=True)

# Separate the features (X) and the target variable (y)
X_features = adult_data.drop('income', axis=1)  # Independent variables (features)
y_target = adult_data['income']                 # Dependent variable (target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier (with a limit on tree depth for simplicity)
classifier_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier_tree.fit(X_train, y_train)

# Visualize the Decision Tree and save it as an image
plt.figure(figsize=(12, 7))
tree.plot_tree(classifier_tree, filled=True, rounded=True, feature_names=X_features.columns, class_names=['<=50K', '>50K'], proportion=False, impurity=False, fontsize=10)

# Save the plot
plt.savefig('decision_tree_output.png')

# Make predictions on the test set
predicted_outcome = classifier_tree.predict(X_test)

# Precision, recall, and support for each class
precision = precision_score(y_test, predicted_outcome, average=None)
recall = recall_score(y_test, predicted_outcome, average=None)
support = confusion_matrix(y_test, predicted_outcome).sum(axis=1)

# Class labels
class_labels = ['<=50K', '>50K']

print("Classification Report:")
print("-" * 40)
for i, label in enumerate(class_labels):
    print(f"Class {label}:")
    print(f"   Precision: {precision[i]:.2f}")
    print(f"   Recall:    {recall[i]:.2f}")
    print(f"   Support:   {support[i]}")
    print("-" * 40)

# Print overall accuracy
accuracy = accuracy_score(y_test, predicted_outcome)
print(f"Accuracy: {accuracy:.2f}")