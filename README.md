# Classification Dataset Analysis

**Track:** AI & ML Track  
**Tools:** Python, Pandas, Scikit-learn, Jupyter Notebook  

## Objective
Understand the structure of a basic classification problem by selecting a dataset and identifying its features, target variable, and possible target classes.

## Dataset Overview
For this task, the **Iris Dataset** (`iris_dataset.csv`) is used. It contains physical measurements of iris flowers across different species.

## Feature and Target Identification

### Features (Independent Variables)
The input measurements used to predict the flower species:
* `sepal length (cm)`
* `sepal width (cm)`
* `petal length (cm)`
* `petal width (cm)`

### Target Variable (Dependent Variable)
The categorical output column to be predicted:
* `species`

### Target Classes
The unique categorical values present in the target variable:
1. `setosa`
2. `versicolor`
3. `virginica`

## Explanation of the Classification Problem
* **Problem Type:** Multiclass Classification (the target variable consists of 3 distinct classes).
* **Goal:** The objective of this classification problem is to analyze the four numerical flower measurements (features) and accurately categorize each flower sample into one of the three iris species (target classes).

## Code Implementation

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('iris_dataset.csv')

# Display basic dataset information
print("--- First 5 Rows ---")
print(df.head())

# Identify target classes
target_column = 'species'
unique_classes = df[target_column].unique()

print(f"\nTarget Column: {target_column}")
print(f"Unique Target Classes: {list(unique_classes)}")
print(f"Total Classes: {len(unique_classes)}")

# Determine classification type
problem_type = "Binary Classification" if len(unique_classes) == 2 else "Multiclass Classification"
print(f"Problem Type: {problem_type}")
```

## Interview Questions and Answers

### 1. What is classification?
**Answer:** Classification is a type of supervised machine learning where the goal is to predict the categorical class label of a given observation or data point based on input features. Common examples include email spam detection and medical diagnosis.

### 2. What is binary classification?
**Answer:** Binary classification is a classification task where the target variable has exactly two possible mutually exclusive outcomes or classes (for example, Yes/No, True/False, 0/1, or Spam/Not Spam).

### 3. What is multiclass classification?
**Answer:** Multiclass classification is a classification task where the target variable has three or more possible classes, and each observation is assigned to one and only one of those classes (for example, classifying iris flowers into `setosa`, `versicolor`, or `virginica`).
