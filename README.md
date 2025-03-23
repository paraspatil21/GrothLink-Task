# **Car Purchase Prediction Using Machine Learning**

## **Project Overview**

This project focuses on predicting car purchase amounts using machine learning techniques. It involves data preprocessing, feature scaling, outlier removal, and training a **Random Forest Regressor** model.

## **Technologies Used**

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Chardet

## **Dataset Information**

- The dataset contains customer details and their car purchase amounts.
- It includes attributes such as income, age, credit card debt, and net worth.
- The dataset undergoes preprocessing to handle missing values and outliers.

## **Installation & Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`car_purchasing.csv`) in the project directory.

## **Data Preprocessing**

- Detects and applies the correct encoding using `chardet`.
- Handles missing values by filling them with the median.
- Removes outliers using the **Z-score method**.
- Drops non-relevant columns like customer name, email, and country.
- Splits data into training (80%) and testing (20%) sets.
- Applies feature scaling using `StandardScaler`.

## **Model Training & Evaluation**

- Uses a **Random Forest Regressor** with 100 estimators.
- Evaluates the model using:
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Visualizes actual vs. predicted sales using a scatter plot.

## **Results & Insights**

- A well-trained model should exhibit a **linear trend** between actual and predicted sales.
- A high **R² Score** indicates better model accuracy.
- The scatter plot helps assess prediction performance.

## **Usage**

Run the script:
```bash
python car_purchase_prediction.py
```

## **Future Enhancements**

- Experiment with different regression models (XGBoost, Gradient Boosting, etc.).
- Perform hyperparameter tuning for better accuracy.
- Deploy the model using Flask or FastAPI for real-world usage.

## **Author**

- **Your Name**
- Email: your.email@example.com
- GitHub: [your-profile](https://github.com/your-profile)
