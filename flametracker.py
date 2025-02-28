import os
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import nbformat as nbf

def create_jupyter_notebook():
    """Generates a Jupyter Notebook version of this script with separate execution blocks."""
    notebook = nbf.v4.new_notebook()
    
    # Markdown introduction
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    # Predicting Burnt Area During Forest Fires
    This notebook analyzes and predicts the burnt area during forest fire incidents using machine learning.
    """))
    
    # Install required packages
    notebook.cells.append(nbf.v4.new_code_cell("""
    !pip install pandas kagglehub matplotlib seaborn scikit-learn numpy
    """))
    
    # Import libraries
    notebook.cells.append(nbf.v4.new_code_cell("""
import os
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
"""))
    
    # Download Dataset
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    ## Download Dataset
    Downloads the latest version of the dataset using KaggleHub.
    """))
    notebook.cells.append(nbf.v4.new_code_cell("""
dataset_path = kagglehub.dataset_download("sumitm004/forest-fire-area")
print(f"Dataset downloaded to: {dataset_path}")
"""))
    
    # Load Dataset
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    ## Load Dataset
    Loads the dataset and preprocesses categorical variables.
    """))
    notebook.cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv(os.path.join(dataset_path, "forestfires.csv"))
df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)
df.head()
"""))
    
    # Preprocess Data
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    ## Preprocess Data
    Standardizes numerical features using StandardScaler.
    """))
    notebook.cells.append(nbf.v4.new_code_cell("""
scaler = StandardScaler()
numerical_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
df[numerical_features] = scaler.fit_transform(df[numerical_features])
df.head()
"""))
    
    # Feature Engineering
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    ## Feature Engineering
    Creates new features such as log-transformed area and fire potential.
    """))
    notebook.cells.append(nbf.v4.new_code_cell("""
df['log_area'] = np.log1p(df['area'])
df['fire_potential'] = df['temp'] * df['wind']
df.head()
"""))
    
    # Exploratory Data Analysis
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    ## Exploratory Data Analysis
    Generates a correlation heatmap and burnt area distribution plots.
    """))
    notebook.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(14, 10))  # Increase figure size
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8}, linewidths=0.5)
plt.yticks(fontsize=10)  # Reduce y-axis label size
plt.title("Feature Correlation Heatmap", fontsize=12, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['area'], bins=100, kde=True)
plt.title("Burnt Area Distribution")
plt.xlabel("Burnt Area (ha)")
plt.ylabel("Frequency")
plt.xlim(0, df['area'].quantile(0.99))
plt.show()
"""))
    
    # Train and Evaluate Model
    notebook.cells.append(nbf.v4.new_markdown_cell("""
    ## Train and Evaluate Model
    Trains a Random Forest model with hyperparameter tuning and evaluates performance.
    """))
    notebook.cells.append(nbf.v4.new_code_cell("""
features = df.drop(columns=['area', 'log_area'])
target = df['log_area']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': best_model.feature_importances_})
print("Top Features:")
print(feature_importances.sort_values(by='Importance', ascending=False).head(10))
"""))
    
    # Save as a Jupyter notebook file
    with open("Forest_Fire_Prediction.ipynb", "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print("Jupyter Notebook created: Forest_Fire_Prediction.ipynb")

# Run the function to generate the notebook
create_jupyter_notebook()
