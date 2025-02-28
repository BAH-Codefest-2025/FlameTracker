# 🔥 FlameTracker: Predicting Burnt Area in Forest Fires

## 🌍 Challenge 2: Flame Forecast

### 📌 Overview
**FlameTracker** is a machine learning project aimed at predicting the burnt area during forest fire incidents. Forest fires pose a significant environmental threat, and understanding the conditions that contribute to severe wildfires can help in better fire management and prevention strategies.

---

## 🎯 Goals and Objectives
- **Predict** the burnt area of a forest fire based on environmental and meteorological conditions.
- **Analyze** the impact of different factors like temperature, humidity, wind, and rain on fire severity.
- **Enhance** fire prevention strategies using data-driven insights.

---

## 📂 Dataset
- This dataset contains records of **517** forest fires from the Montesinho Natural Park in Portugal. Each fire incident includes details such as the day of the week, month, and geographic coordinates, along with the burnt area (in hectares). Additionally, it provides several meteorological variables, including rainfall, temperature, humidity, and wind speed, which are critical factors influencing fire behavior.
- Dataset Source: [https://www.kaggle.com/competitions/forestfiresarea/overview](https://www.kaggle.com/datasets/sumitm004/forest-fire-area/data)

---

## 🛠 Implementation Details
This project was implemented using Python and the following **libraries**:
- **Data Handling & Processing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Dataset Handling:** `kagglehub`

### 🔧 Key Features
- **Data Preprocessing:** Normalization using `StandardScaler`
- **Feature Engineering:** New features like `fire_potential` and `log_area`
- **Exploratory Data Analysis (EDA):** Heatmaps and distributions to understand correlations
- **Machine Learning Model:** Hyperparameter-tuned `RandomForestRegressor`
- **Performance Evaluation:** `MAE`, `MSE`, `R² Score`

---

## 🚀 How to Use
### 📥 Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/flametracker.git
   cd flametracker
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure you have **Kaggle API** configured to download the dataset.

### ▶️ Running the Project
Run the main script to execute the full pipeline:
```sh
python flametracker.py
```


## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
Feel free to **fork** this repository, create a **feature branch**, and submit a **pull request**!

---
