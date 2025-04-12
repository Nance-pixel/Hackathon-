# Import libraries with error handling
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as e:
    print(f"Error: Missing library. Please install it using: pip install {str(e).split('No module named ')[1]}")
    exit(1)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Preparation
try:
    # Load Iris dataset (replace with your dataset, e.g., pd.read_csv('your_data.csv'))
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    print(f"Error in data preparation: {e}")
    exit(1)

# 2. Model Training and Hyper-Parameter Tuning
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': [
            {'C': [0.1, 1], 'solver': ['lbfgs'], 'penalty': ['l2']},
            {'C': [0.1, 1], 'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
            {'C': [0.1, 1], 'solver': ['saga'], 'penalty': ['l1', 'l2']}
        ]
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {'max_depth': [3, 5   ], 'min_samples_split': [2]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [50], 'max_depth': [5]}
    },
    'Support Vector Machine': {
        'model': SVC(),
        'params': {'C': [1], 'kernel': ['rbf']}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(),
        'params': {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3]}
    }
}

results = []

# Train and tune models
for model_name, config in models.items():
    print(f"Training {model_name}...")
    try:
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise'
        )
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        avg_type = 'binary' if len(np.unique(y)) == 2 else 'weighted'
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'R2 Score': r2,
            'Best Parameters': grid_search.best_params_
        })
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        continue

# Convert results to DataFrame
try:
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("Error: No models were successfully trained.")
        exit(1)
except Exception as e:
    print(f"Error creating results DataFrame: {e}")
    exit(1)

# 3. Dashboard Creation
try:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score', 'R2 Score'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, None]]
    )

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'R2 Score']
    for i, metric in enumerate(metrics, 1):
        row = 1 if i <= 3 else 2
        col = i if i <= 3 else i - 3
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df[metric], name=metric),
            row=row, col=col
        )

    fig.update_layout(
        title_text="Model Performance Comparison Dashboard",
        height=800,
        width=1200,
        showlegend=False
    )

    fig.show()
    fig.write_html('model_performance_dashboard.html')
    print("Dashboard saved as 'model_performance_dashboard.html'")
except Exception as e:
    print(f"Error creating dashboard: {e}")

# Print results
try:
    print("\nModel Performance Summary:")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'R2 Score']].round(3))
    print("\nBest Parameters:")
    for index, row in results_df.iterrows():
        print(f"{row['Model']}: {row['Best Parameters']}")
except Exception as e:
    print(f"Error printing results: {e}")

# 4. Additional Visualizations
try:
    # Correlation table (instead of seaborn heatmap)
    print("\nFeature Correlation Table:")
    corr_matrix = X.corr().round(3)
    print(corr_matrix)

    # Feature importance for Random Forest using matplotlib
    rf_model = models['Random Forest']['model'].fit(X_train_scaled, y_train)
    feature_importance = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'")
except Exception as e:
    print(f"Error creating visualizations: {e}")
