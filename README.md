# Sales Forecasting with Multiple Linear Regression

This project implements a multiple linear regression model for sales forecasting using comprehensive business metrics. It analyzes various factors affecting sales performance to predict future revenue.

## Features

The model uses the following input features:
- Marketing Metrics:
  - marketing_spend: Total marketing budget
  - website_traffic: Number of website visitors
  - lead_conversion_rate: Rate of converting leads to customers
- Sales Performance:
  - sales_calls: Number of sales calls made
  - proposal_submissions: Number of proposals submitted
  - new_clients: Number of new clients acquired
- Customer Metrics:
  - customer_retention_rate: Rate of customer retention
  - recurring_revenue: Revenue from existing customers
- Market Conditions:
  - gdp_growth: GDP growth rate
  - inflation_rate: Current inflation rate
  - it_industry_growth: IT industry growth rate
  - competitor_pricing: Average competitor pricing
- Operational Metrics:
  - sales_cycle_duration: Average duration of sales cycle
  - csat_score: Customer satisfaction score

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sales-forecasting
```

2. Create and activate a virtual environment:

On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Data Format

The input data should be a CSV file (`sales_data.csv`) with the following columns:
- marketing_spend: Marketing budget in dollars
- website_traffic: Number of website visitors
- lead_conversion_rate: Conversion rate (0-1)
- sales_calls: Number of sales calls
- proposal_submissions: Number of proposals
- new_clients: Number of new clients
- customer_retention_rate: Retention rate (0-1)
- recurring_revenue: Revenue from existing customers
- gdp_growth: GDP growth percentage
- inflation_rate: Inflation rate percentage
- it_industry_growth: IT industry growth percentage
- competitor_pricing: Average competitor price
- sales_cycle_duration: Sales cycle duration in days
- csat_score: Customer satisfaction score (1-5)
- Sales: Target variable (total sales revenue)

## Usage

Run the main script:
```bash
python sales_forecasting.py
```

The script will:
1. Load and preprocess the data
2. Select the most important features (top 8)
3. Train the multiple linear regression model
4. Evaluate the model's performance
5. Generate visualizations
6. Make predictions for future sales

## Output

1. Model Performance Metrics:
   - R² Score: How well the model fits the data
   - Root Mean Squared Error: Average prediction error
   - Mean Absolute Error: Average absolute prediction error
   - Feature importance scores

2. Visualizations (`sales_forecasting_results.png`):
   - Actual vs Predicted Sales: Model accuracy visualization
   - Residual Plot: Error distribution analysis
   - Feature Importance: Impact of each feature
   - Correlation Matrix: Relationships between variables

3. Future Sales Predictions:
   - Example predictions for sample future scenarios

## Customization

To use your own data:
1. Replace `sales_data.csv` with your data file
2. Ensure your data follows the required format
3. Adjust the number of features to select in the `select_features` method
4. Modify the future data in the `main` function to match your needs

## Model Interpretation

The model provides:
- Feature importance scores to identify the most influential factors
- Correlation matrix to understand relationships between variables
- Residual plot to check model assumptions
- Actual vs Predicted plot to visualize prediction accuracy 