# Traffic Congestion Prediction Using Machine Learning

This repository contains the code to predict traffic congestion levels (High, Medium, or Low) for road sections using traffic sensor data. The project leverages a Random Forest Classifier to classify traffic congestion levels based on the available data.

## Project Overview

- **Input:** Traffic sensor data stored in a CSV file. The dataset contains various features (e.g., vehicle count, speed, time of day) that can be used to predict the congestion level for a road section.
- **Output:** Predicted congestion level (High, Medium, Low) for a given set of input features.
- **Model:** Random Forest Classifier, trained on the input features to classify the target column (traffic congestion).

## Requirements

Ensure the following libraries are installed:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn google-colab

Steps to Use the Code
Step 1: Import Required Libraries
The necessary Python libraries are imported to handle data manipulation, visualization, machine learning, and model evaluation.

python
Copy
Edit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

## Step 2: Upload Your CSV File
Use the file upload functionality to upload your dataset (CSV format) containing traffic sensor data.

python
Copy
Edit
from google.colab import files
uploaded = files.upload()
 ## Step 3: Load the Dataset
After uploading the file, the dataset is loaded into a pandas DataFrame for processing.

python
Copy
Edit
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
## Step 4: Clean Column Names
Clean the column names by removing any leading or trailing spaces that may have been inadvertently included.

python
Copy
Edit
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
## Step 5: Detect Target Column
Automatically detect the column that contains the congestion levels. If no column is found, you must manually identify the target column.

python
Copy
Edit
possible_target_cols = [col for col in df.columns if 'congestion' in col.lower()]
## Step 6: Encode the Target Column
The target column (congestion level) is encoded into numeric values (0, 1, 2) using LabelEncoder.

python
Copy
Edit
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])
## Step 7: Define Features and Target Variables
Separate the features (X) and the target variable (y).

python
Copy
Edit
X = df.drop(target_col, axis=1)
y = df[target_col]
If necessary, handle non-numeric features by applying one-hot encoding.

python
Copy
Edit
X = pd.get_dummies(X)
## Step 8: Train/Test Split
Split the dataset into training and testing sets (80% for training and 20% for testing).

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Step 9: Train the Model
Train the Random Forest Classifier on the training data.

python
Copy
Edit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
## Step 10: Evaluate the Model
Evaluate the trained model using various metrics such as accuracy, precision, and recall.

python
Copy
Edit
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
Generate a classification report and confusion matrix to assess the performance.

python
Copy
Edit
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
## Step 11: Visualize Confusion Matrix
Plot a heatmap of the confusion matrix to visually assess the classifier's performance.

python
Copy
Edit
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
## Step 12: Predict on New Data
Enter values for a new road section and predict the congestion level using the trained model.

python
Copy
Edit
sample_input = []  # Collect user input for prediction
sample_pred = model.predict(sample_df)[0]
Example Output
When a new sample is entered, the model will output the predicted congestion level (High, Medium, or Low).

bash
Copy
Edit
🟢 Predicted Congestion Level: Medium
# Conclusion
This project demonstrates how machine learning can be used to predict traffic congestion levels based on sensor data. The Random Forest model can be further tuned and improved by adjusting hyperparameters or using a different machine learning algorithm depending on the dataset's complexity and the accuracy required.


