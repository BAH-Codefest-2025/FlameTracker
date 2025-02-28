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

def download_dataset(dataset_name="sumitm004/forest-fire-area"):
    """Downloads the latest version of the dataset using KaggleHub."""
    path = kagglehub.dataset_download(dataset_name)
    return path

def load_dataset(path, filename="forestfires.csv"):
    """Loads the dataset into a Pandas DataFrame and preprocesses it."""
    dataset_path = os.path.join(path, filename)
    df = pd.read_csv(dataset_path)
    df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)
    return df

def preprocess_data(df):
    """Scales numerical features using StandardScaler to normalize the data."""
    scaler = StandardScaler()
    numerical_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def feature_engineering(df):
    """Creates new features and applies transformations to improve model accuracy."""
    df['log_area'] = np.log1p(df['area'])
    df['fire_potential'] = df['temp'] * df['wind']
    return df

def explore_data(df):
    """Performs data exploration including feature correlations and burnt area distribution visualization."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.show()
    plot_burnt_area_distribution(df)

def plot_burnt_area_distribution(df):
    """Plots the distribution of burnt area with a more spread-out x-axis."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df['area'], bins=100, kde=True)
    plt.title("Burnt Area Distribution")
    plt.xlabel("Burnt Area (ha)")
    plt.ylabel("Frequency")
    plt.xlim(0, df['area'].quantile(0.99))
    plt.show()

def train_test_model(df):
    """Trains a Random Forest model with hyperparameter tuning and evaluates performance."""
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
    
    print("Predict the burnt area during the forest fire incidents based on the given conditions.")
    print("Model Performance (Tuned Random Forest):")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': best_model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print("\nTop Features:")
    print(feature_importances.head(10))
    
    return best_model

def main():
    dataset_path = download_dataset()
    df = load_dataset(dataset_path)
    explore_data(df)
    df = preprocess_data(df)
    df = feature_engineering(df)
    train_test_model(df)

if __name__ == "__main__":
    main()
