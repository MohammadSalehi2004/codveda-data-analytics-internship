# Codveda Data Analytics Internship

This repository contains tasks completed as part of the Codveda Data Analytics Internship. The focus of this project is to apply data analysis techniques using Python and gain some experience by doing tasks and getting a feel of how a real world role would look like.

---

## Level 1 Tasks

### 🔹 Task 1: Data Cleaning and Preprocessing

* Loaded raw dataset using pandas
* Handled missing values (removal/imputation)
* Removed duplicate records
* Standardized inconsistent data formats
* Exported cleaned dataset for further analysis

---

### 🔹 Task 3: Basic Data Visualization

* Created visualizations such as:

  * Line charts
  * Bar plots
  * Scatter plots
* Customized titles, labels, and legends
* Exported plots as images for reporting

Output:
- price_trend.png  
- monthly_avg_price.png  
- open_vs_close.png

---

## Level 2 Tasks

### 🔹 Task 1: Regression Analysis

* Used House Prediction dataset but the cells were all messed up so I had AI fix it(all the columns were in column A)
* Performed data inspection and basic cleaning (handling missing values like level 1 task 1)
* Split dataset into training 80% and testing 20% sets  
* Applied Linear Regression using scikit-learn  
* Evaluated model using:
  * R^2 Score to find how well the model predicts
  * Mean Squared Error (MSE) to find how well the model makes mistakes
* Interpreted model coefficients
* Visualized actual vs predicted values using a scatter plot

Output:
- regression_plot.png 

### 🔹 Task 3: Clustering Analysis (K-Means)

* Used Iris dataset  
* Removed the last column called species since it was in text form and not numerical
* Standardized features using StandardScaler  
* Applied K-Means clustering
* Set optimal number of clusters using the Elbow Method  
* Visualized clusters using 2D scatter plot

Output:
- elbow_method.png  
- iris_clusters.png  
---
## Tools

* Python
* pandas
* matplotlib
* seaborn
* scikit-learn

---

## Project Structure

```
codveda-data-analytics-internship/
│
├── Data/
│   ├── stock_prices.csv
│   ├── cleaned_stock_prices.csv
│   ├── house_Prediction_Data_Set.csv
│   └── iris.csv
│
├── Level1/
│   ├── task1_data_cleaning.py
│   ├── task3_visualization.py
│   └── Plots/
│       ├── price_trend.png
│       ├── monthly_avg_price.png
│       └── open_vs_close.png
│
├── Level2/
│   ├── task1_regression.py
│   ├── task3_clustering.py
│   └── Plots/
│       ├── regression_plot.png
│       ├── elbow_method.png
│       └── iris_clusters.png
│
├── README.md
└── requirements.txt
```
## Visualizations of Plots folder

### Stock Price Trend
![Price Trend](Level1/Plots/price_trend.png)

### Average Monthly Closing Price
![Monthly Average](Level1/Plots/monthly_avg_price.png)

### Open vs Close Price
![Scatter Plot](Level1/Plots/open_vs_close.png)

---

### Regression Plot (Actual vs Predicted Prices)
![Regression Plot](Level2/Plots/regression_plot.png)

### Elbow Method (Optimal Clusters)
![Elbow Method](Level2/Plots/elbow_method.png)

### Iris Clusters Visualization
![Iris Clusters](Level2/Plots/iris_clusters.png)
---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/yourusername/codveda-data-analytics-internship.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the codes inside each level folder so that it works

---

## Dataset

The datasets used in this project include stock price data, house prediction data, and the iris dataset, and are used for:

* Data cleaning and preprocessing
* Trend analysis and visualization
* Regression analysis
* Clustering analysis
* They are provided inside the Data folder for the internship

---

## Reflection

* For Level 1, I learned how to stregthen my skills regarding datsets by improving my understanding of handling missing values, formatting data, and presenting insights visually and how to plot and visualize graphs in a cleaner way.
* In Level 2, I learned how to apply machine learning techniques such as Linear Regression and K-Means clustering in a more professional manner. I now gained experience in evaluating models using R² score and learned to use MSE and visualizing results to understand model performance.
---

## Author

**Mohammad Salehi**
Student Learning Assistant – Information Technology year 1 at Middlesex University Dubai

---

## Notes

This repository is part of an internship project and will be updated as more tasks are completed.
