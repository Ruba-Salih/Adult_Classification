#!/usr/bin/python3
"""Modified Decision Tree Classifier for the Adult Census Dataset."""

import pandas as pd
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
# Replace 'adult.csv' with your dataset's path
adult_data = pd.read_csv('adult.csv')

# Step 2: Handle missing values (numeric and categorical features separately)

# Define the numerical and categorical feature columns
numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_nominal = ['workclass', 'occupation', 'native.country', 'marital.status', 'relationship', 'race', 'sex']
categorical_ordinal = ['education']

# Impute missing values for numerical columns (using the median)
median_imputer = SimpleImputer(strategy='median')
adult_data[numerical_features] = median_imputer.fit_transform(adult_data[numerical_features])

# Impute missing values for nominal categorical columns (using the most frequent value)
mode_imputer_nominal = SimpleImputer(strategy='most_frequent')
adult_data[categorical_nominal] = mode_imputer_nominal.fit_transform(adult_data[categorical_nominal])

# Impute missing values for ordinal categorical columns (using the most frequent value)
mode_imputer_ordinal = SimpleImputer(strategy='most_frequent')
adult_data[categorical_ordinal] = mode_imputer_ordinal.fit_transform(adult_data[categorical_ordinal])

# Step 3: Encode ordinal categorical features (education)
# Define the order for education categories from least to most advanced
education_levels = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 
    'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Masters', 
    'Prof-school', 'Doctorate'
]

# Apply Ordinal Encoding to the 'education' feature
ordinal_encoder = OrdinalEncoder(categories=[education_levels])
adult_data['education'] = ordinal_encoder.fit_transform(adult_data[['education']])

# Step 4: One-Hot Encoding for nominal categorical features
adult_data = pd.get_dummies(adult_data, columns=categorical_nominal, drop_first=True)

# Step 5: Separate the features (X) and the target variable (y)
X_features = adult_data.drop('income', axis=1)  # Independent variables (features)
y_target = adult_data['income']                 # Dependent variable (target)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Step 7: Train a Decision Tree Classifier (with a limit on tree depth for simplicity)
classifier_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier_tree.fit(X_train, y_train)

# Step 8: Visualize the Decision Tree and save it as an image
plt.figure(figsize=(12, 7))
tree.plot_tree(classifier_tree, filled=True, rounded=True, feature_names=X_features.columns, class_names=['<=50K', '>50K'], proportion=False, impurity=False, fontsize=10)

# Save the plot
plt.savefig('decision_tree_output.png')
print("Decision tree plot saved as 'decision_tree_output.png'")

# Step 9: Make predictions on the test set
predicted_outcome = classifier_tree.predict(X_test)

# Step 10: Evaluate the model's performance
accuracy = accuracy_score(y_test, predicted_outcome)
print(f"Model Accuracy: {accuracy:.2f}")

# Print detailed classification report
performance_report = classification_report(y_test, predicted_outcome)
print(performance_report)
