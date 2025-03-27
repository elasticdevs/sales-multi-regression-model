import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime

class SalesForecaster:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, file_path):
        """
        Load and preprocess the sales data.
        Args:
            file_path: Path to the CSV file containing sales data
        Returns:
            X: Feature matrix
            y: Target variable (sales)
        """
        # Load the data
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time-based features
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(df.info())
        print("\nSample of the data:")
        print(df.head())
        
        # Separate features and target
        X = df.drop(['Sales', 'date'], axis=1)
        y = df['Sales']
        
        return X, y, df['date']
    
    def select_features(self, X, y, k=8):
        """
        Select the most important features using f_regression.
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
        """
        # Remove non-numeric columns before feature selection
        numeric_cols = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
        X_numeric = X[numeric_cols]
        
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.feature_selector.fit_transform(X_numeric, y)
        self.selected_features = numeric_cols[self.feature_selector.get_support()].tolist()
        
        # Display feature importance scores
        feature_scores = pd.DataFrame({
            'Feature': numeric_cols,
            'Score': self.feature_selector.scores_
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        print("\nFeature Importance Scores:")
        print(feature_scores)
        
        return X_selected
    
    def train_model(self, X, y):
        """
        Train the multiple linear regression model.
        Args:
            X: Feature matrix
            y: Target variable
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate the model
        self.evaluate_model(y_test, y_pred)
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate the model's performance.
        Args:
            y_true: Actual values
            y_pred: Predicted values
        """
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        print("\nModel Performance Metrics:")
        print(f"R² Score: {r2:.4f}")
        print(f"Root Mean Squared Error: ${rmse:,.2f}")
        print(f"Mean Absolute Error: ${mae:,.2f}")
        
        # Print model coefficients
        print("\nModel Coefficients:")
        for feature, coef in zip(self.selected_features, self.model.coef_):
            print(f"{feature}: ${coef:,.2f}")
        print(f"Intercept: ${self.model.intercept_:,.2f}")
    
    def plot_results(self, X, y, X_test, y_test, dates):
        """
        Create visualizations of the model results.
        Args:
            X: Training features
            y: Training target
            X_test: Test features
            y_test: Test target
            dates: Date series
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 20))
        
        # Plot 1: Actual vs Predicted
        plt.subplot(3, 2, 1)
        y_pred = self.model.predict(self.scaler.transform(X_test))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Sales ($)')
        plt.ylabel('Predicted Sales ($)')
        plt.title('Actual vs Predicted Sales')
        
        # Plot 2: Residuals
        plt.subplot(3, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Sales ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residual Plot')
        
        # Plot 3: Feature Importance
        plt.subplot(3, 2, 3)
        importance = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': abs(self.model.coef_)
        })
        importance = importance.sort_values('Importance', ascending=True)
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Absolute Coefficient Value')
        plt.title('Feature Importance')
        
        # Plot 4: Correlation Matrix
        plt.subplot(3, 2, 4)
        numeric_cols = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
        data = pd.concat([X[numeric_cols], y], axis=1)
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        
        # Plot 5: Sales Trend Over Time
        plt.subplot(3, 2, 5)
        plt.plot(dates, y, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Sales ($)')
        plt.title('Sales Trend Over Time')
        plt.xticks(rotation=45)
        
        # Plot 6: Monthly Sales Distribution
        plt.subplot(3, 2, 6)
        monthly_sales = pd.DataFrame({
            'date': dates,
            'sales': y
        })
        monthly_sales['month'] = monthly_sales['date'].dt.month_name()
        sns.boxplot(x='month', y='sales', data=monthly_sales)
        plt.xticks(rotation=45)
        plt.title('Monthly Sales Distribution')
        
        plt.tight_layout()
        plt.savefig('sales_forecasting_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nVisualization saved as 'sales_forecasting_results.png'")
    
    def predict_future_sales(self, future_data):
        """
        Make predictions for future sales.
        Args:
            future_data: DataFrame containing future feature values
        Returns:
            Predicted sales values
        """
        # Select only the numeric features used in training
        future_data_selected = future_data[self.selected_features]
        
        # Scale the features
        future_data_scaled = self.scaler.transform(future_data_selected)
        
        # Make predictions
        predictions = self.model.predict(future_data_scaled)
        
        return predictions

def main():
    # Initialize the forecaster
    forecaster = SalesForecaster()
    
    # Load the data
    print("Loading data...")
    X, y, dates = forecaster.load_data('sales_data.csv')
    
    # Select important features
    print("\nSelecting important features...")
    X_selected = forecaster.select_features(X, y)
    
    # Train the model
    print("\nTraining the model...")
    X_train, X_test, y_train, y_test = forecaster.train_model(X_selected, y)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    forecaster.plot_results(X, y, X_test, y_test, dates)
    
    # Example: Predict future sales
    print("\nMaking future predictions...")
    future_data = pd.DataFrame({
        'marketing_spend': [65000, 70000, 75000],
        'website_traffic': [21000, 22000, 23000],
        'lead_conversion_rate': [0.23, 0.24, 0.25],
        'sales_calls': [145, 150, 155],
        'proposal_submissions': [42, 45, 48],
        'new_clients': [12, 13, 14],
        'customer_retention_rate': [0.93, 0.94, 0.95],
        'recurring_revenue': [92000, 94000, 96000],
        'gdp_growth': [3.4, 3.5, 3.6],
        'inflation_rate': [2.5, 2.6, 2.7],
        'it_industry_growth': [4.1, 4.2, 4.3],
        'competitor_pricing': [1350, 1400, 1450],
        'sales_cycle_duration': [35, 34, 33],
        'csat_score': [4.9, 5.0, 5.0],
        'month': [4, 5, 6],
        'year': [2024, 2024, 2024],
        'quarter': [2, 2, 2]
    })
    
    predictions = forecaster.predict_future_sales(future_data)
    print("\nPredicted Sales for Future Data:")
    for i, pred in enumerate(predictions):
        print(f"Case {i+1}: ${pred:,.2f}")

if __name__ == "__main__":
    main() 