#!/usr/bin/python3
"""This is the base model class for AirBnB"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset (replace with your local path)
data = pd.read_csv('path_to_your_file/adult.csv')

# Step 2: Handle missing values

# Define the columns that are numerical and categorical
num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols_nominal = ['workclass', 'occupation', 'native-country']  # Nominal categorical columns
cat_cols_ordinal = ['education']  # Ordinal categorical column with inherent order

# Imputer for numerical columns (filling with the median)
imputer_num = SimpleImputer(strategy='median')
data[num_cols] = imputer_num.fit_transform(data[num_cols])

# Imputer for nominal categorical columns (filling with the most frequent value - mode)
imputer_cat_nominal = SimpleImputer(strategy='most_frequent')
data[cat_cols_nominal] = imputer_cat_nominal.fit_transform(data[cat_cols_nominal])

# Imputer for ordinal categorical columns (filling with the most frequent value - mode)
imputer_cat_ordinal = SimpleImputer(strategy='most_frequent')
data[cat_cols_ordinal] = imputer_cat_ordinal.fit_transform(data[cat_cols_ordinal])

# Step 3: Ordinal Encoding for education (ordered categories)
# Define the ordered categories for education
education_categories = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 
    'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'
]

# Apply Ordinal Encoding
encoder_ordinal = OrdinalEncoder(categories=[education_categories])
data['education'] = encoder_ordinal.fit_transform(data[['education']])

# Step 4: One-Hot Encoding for the remaining nominal categorical variables
data = pd.get_dummies(data, columns=cat_cols_nominal, drop_first=True)

# Step 5: Split the data into features (X) and target variable (y)
X = data.drop('income', axis=1)  # Features (input data)
y = data['income']               # Target (what we want to predict)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Predict on the test set
y_pred = clf.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print(classification_report(y_test, y_pred))
