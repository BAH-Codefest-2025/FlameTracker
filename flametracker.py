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
    
    # Convert categorical variables (month, day) into dummy/indicator variables
    df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)
    return df

def preprocess_data(df):
    """Scales numerical features using StandardScaler to normalize the data."""
    scaler = StandardScaler()
    numerical_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    
    # Apply standard scaling to numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def feature_engineering(df):
    """Creates new features and applies transformations to improve model accuracy."""
    
    # Log-transform the target variable to reduce skewness
    df['log_area'] = np.log1p(df['area'])
    
    # Create a new feature representing fire potential based on temperature and wind speed
    df['fire_potential'] = df['temp'] * df['wind']
    return df

def explore_data(df, save_path="output_graphs"):
    """Performs data exploration including feature correlations and burnt area distribution visualization."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create directory if it doesn't exist

    # Generate and save a heatmap to visualize feature correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    heatmap_path = os.path.join(save_path, "correlation_heatmap.png")
    plt.savefig(heatmap_path)  # Save the figure
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")

    # Plot and save the burnt area distribution
    plot_burnt_area_distribution(df, save_path)

def plot_burnt_area_distribution(df, save_path="output_graphs"):
    """Plots the distribution of burnt area and saves the plot."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df['area'], bins=100, kde=True)
    plt.title("Burnt Area Distribution")
    plt.xlabel("Burnt Area (ha)")
    plt.ylabel("Frequency")
    
    # Limit x-axis to exclude extreme outliers and enhance visualization
    plt.xlim(0, df['area'].quantile(0.99))

    # Save the plot as an image file
    distribution_path = os.path.join(save_path, "burnt_area_distribution.png")
    plt.savefig(distribution_path)
    plt.close()
    print(f"Burnt area distribution saved to {distribution_path}")

def train_test_model(df):
    """Trains a Random Forest model with hyperparameter tuning and evaluates performance."""
    
    # Define features and target variable
    features = df.drop(columns=['area', 'log_area'])
    target = df['log_area']
    
    # Split the dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Define a hyperparameter grid for tuning the RandomForest model
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    
    # Perform grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Make predictions using the trained model
    y_pred = best_model.predict(X_test)
    
    # Display model performance metrics
    print("Predict the burnt area during the forest fire incidents based on the given conditions.")
    print("Model Performance (Tuned Random Forest):")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    # Display feature importance rankings
    feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': best_model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print("\nTop Features:")
    print(feature_importances.head(10))
    
    return best_model

def main():
    """Main function to orchestrate dataset loading, exploration, preprocessing, feature engineering, and model training."""
    
    # Download the dataset
    dataset_path = download_dataset()
    
    # Load the dataset into a Pandas DataFrame
    df = load_dataset(dataset_path)
    
    # Define where the images should be saved
    save_path = "output_graphs"
    
    # Perform exploratory data analysis and save images
    explore_data(df, save_path)
    
    # Preprocess numerical features
    df = preprocess_data(df)
    
    # Apply feature engineering techniques
    df = feature_engineering(df)
    
    # Train and evaluate the machine learning model
    train_test_model(df)

if __name__ == "__main__":
    main()
