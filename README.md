# Hackathon-


Iris Model Comparison Dashboard

This project demonstrates the comparison of multiple classification models on the Iris dataset using hyperparameter tuning, evaluation metrics, and visualizations.

Features

Loads and preprocesses the Iris dataset

Trains five different models:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

Gradient Boosting


Performs hyperparameter tuning using GridSearchCV

Evaluates models on:

Accuracy

Precision

Recall

F1-Score

R² Score


Creates an interactive Plotly dashboard comparing model performance

Displays feature importance using a bar plot

Robust error handling throughout


Setup Instructions

1. Clone the Repository

git clone https://github.com/yourusername/iris-model-dashboard.git
cd iris-model-dashboard

2. Install Dependencies

Make sure you have Python 3.7+ installed.

pip install numpy pandas matplotlib scikit-learn plotly

3. Run the Script

python model_comparison.py

This will:

Train all models with hyperparameter tuning

Display a performance dashboard in your browser

Save the dashboard as model_performance_dashboard.html

Save a feature importance plot as feature_importance.png


Files

model_comparison.py: Main script for training, evaluation, and visualization

model_performance_dashboard.html: Output dashboard file (generated)

feature_importance.png: Bar chart of feature importances from the Random Forest model (generated)


Output Example

Interactive bar charts comparing Accuracy, Precision, Recall, F1-Score, and R² Score

Tabular summary of performance metrics

Feature correlation matrix

Feature importance plot


Dataset

This project uses the built-in Iris dataset from scikit-learn. 

License

MIT License


---
