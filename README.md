# Adult Census Income Classification Using Decision Tree

This project implements a **Decision Tree Classifier** to predict whether a person's income exceeds $50K per year based on the UCI Adult Census Income dataset. The project involves preprocessing the dataset, handling missing values, encoding categorical features, and training a decision tree classifier to predict income categories. The trained model is evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**, and the decision tree is visualized and saved as an image file.

## Project Structure

- **adult.csv**: The dataset used for training and testing the classifier.
- **main.py**: The main Python script that handles data preprocessing, model training, evaluation, and decision tree visualization.
- **decision_tree_output.png**: The output file that contains the decision tree visualization after the model is trained.
- **README.md**: This file.

## Dataset

The dataset used is the **UCI Adult Census Income dataset**, which is commonly used for binary classification tasks. The target variable is whether a person earns more than $50K per year or not (`<=50K` or `>50K`). 

The dataset contains both categorical and numerical features such as:
- **age**: The age of the individual.
- **education**: The highest level of education attained.
- **capital.gain**: Income from investment sources, apart from wages/salary.
- **occupation**: The individual's occupation.
- **race**: The race of the individual.
- **sex**: The gender of the individual.
- And many others.

## How It Works

1. **Data Preprocessing**:
   - **Handling Missing Data**: Missing values in numerical features are replaced with the **median**, while missing categorical features are filled with the **most frequent value** (mode).
   - **Encoding Categorical Features**: Categorical variables are encoded using **Ordinal Encoding** for ordered features like education and **One-Hot Encoding** for nominal features like occupation and race.

2. **Model Training**:
   - A **Decision Tree Classifier** is trained to classify whether a person’s income is `<=50K` or `>50K`.
   - The maximum depth of the tree is limited to 3 for simplicity, making the tree easier to visualize and interpret.

3. **Evaluation**:
   - The model's performance is evaluated on the test set using **accuracy**, **precision**, **recall**, and **F1-score**.
   - The model outputs a **classification report** showing these metrics for each income class (`<=50K` and `>50K`).

4. **Decision Tree Visualization**:
   - The trained decision tree is visualized using Matplotlib and Scikit-learn’s `plot_tree` function.
   - The tree structure is saved as an image (`decision_tree_output.png`), displaying the decision rules at each node and the classification outcomes at the leaves.

## Prerequisites

Make sure you have Python 3.x installed and the following Python packages:

- `pandas`
- `scikit-learn`
- `matplotlib`

You can install the required packages using `pip`:

```bash
pip install pandas scikit-learn matplotlib 
```

## Running the Project
To run the project, follow these steps:

- Clone this repository or download the files.
- Make sure the adult.csv dataset is in the same directory as main.py.
- Run the Python script:
``` bash
python3 main.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Ruba Salih