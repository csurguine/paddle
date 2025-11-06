# PADDLE API Documentation

## Data Generation Module

### `DataGenerator`

Class for generating synthetic datasets for various business scenarios.

#### Methods

##### `generate_customer_churn_data(n_samples=10000)`
Generate customer churn dataset for predictive modeling.

**Parameters:**
- `n_samples` (int): Number of customer records to generate

**Returns:**
- `pd.DataFrame`: Customer data with features and churn label

**Example:**
```python
from paddle.data.generator import DataGenerator

generator = DataGenerator(random_state=42)
df = generator.generate_customer_churn_data(n_samples=5000)
```

##### `generate_sales_forecast_data(n_periods=730)`
Generate time series data for sales forecasting.

**Parameters:**
- `n_periods` (int): Number of time periods (days)

**Returns:**
- `pd.DataFrame`: Time series data with sales and temporal features

##### `generate_fraud_detection_data(n_samples=50000)`
Generate transaction data for fraud detection.

**Parameters:**
- `n_samples` (int): Number of transactions

**Returns:**
- `pd.DataFrame`: Transaction data with features and fraud label

##### `generate_product_recommendation_data(n_users=1000, n_products=500)`
Generate user-product interaction data for recommendation systems.

**Parameters:**
- `n_users` (int): Number of users
- `n_products` (int): Number of products

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: (interactions, products)

##### `generate_pricing_optimization_data(n_products=100)`
Generate product pricing and demand data.

**Parameters:**
- `n_products` (int): Number of products

**Returns:**
- `pd.DataFrame`: Product pricing features

## Predictive Models Module

### `PredictiveModels`

Class implementing various predictive modeling techniques.

#### Methods

##### `train_churn_prediction_model(df, target_col='churned')`
Train a customer churn prediction model using Random Forest.

**Parameters:**
- `df` (pd.DataFrame): Customer data
- `target_col` (str): Target column name

**Returns:**
- `dict`: Model performance metrics including accuracy, precision, recall, F1-score

**Example:**
```python
from paddle.models.predictive import PredictiveModels

model = PredictiveModels(random_state=42)
metrics = model.train_churn_prediction_model(df)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

##### `train_sales_forecast_model(df, target_col='sales', forecast_horizon=30)`
Train a sales forecasting model using Gradient Boosting.

**Parameters:**
- `df` (pd.DataFrame): Time series data
- `target_col` (str): Target column name
- `forecast_horizon` (int): Number of periods to forecast

**Returns:**
- `dict`: Model performance metrics including RMSE, MAE, R²

##### `train_fraud_detection_model(df, target_col='is_fraud')`
Train a fraud detection model with class balancing.

**Parameters:**
- `df` (pd.DataFrame): Transaction data
- `target_col` (str): Target column name

**Returns:**
- `dict`: Model performance metrics

##### `predict(model_name, X)`
Make predictions using a trained model.

**Parameters:**
- `model_name` (str): Name of trained model ('churn', 'fraud', 'sales_forecast')
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `np.ndarray`: Predictions

## Prescriptive Models Module

### `PrescriptiveModels`

Class implementing prescriptive analytics for optimization.

#### Methods

##### `optimize_pricing(df, objective='maximize_profit')`
Optimize product pricing to maximize profit or revenue.

**Parameters:**
- `df` (pd.DataFrame): Product pricing data
- `objective` (str): 'maximize_profit' or 'maximize_revenue'

**Returns:**
- `dict`: Optimization results with recommended prices

**Example:**
```python
from paddle.models.prescriptive import PrescriptiveModels

model = PrescriptiveModels()
results = model.optimize_pricing(df, objective='maximize_profit')
print(f"Profit increase: ${results['total_optimal_profit'] - results['total_current_profit']:,.2f}")
```

##### `optimize_product_recommendations(interactions_df, products_df, user_id, n_recommendations=10)`
Generate personalized product recommendations.

**Parameters:**
- `interactions_df` (pd.DataFrame): User-product interactions
- `products_df` (pd.DataFrame): Product features
- `user_id` (int): User ID
- `n_recommendations` (int): Number of recommendations

**Returns:**
- `list`: Recommended products with scores

##### `optimize_inventory_allocation(df, total_budget=100000)`
Optimize inventory allocation based on ROI and budget.

**Parameters:**
- `df` (pd.DataFrame): Product data
- `total_budget` (float): Available budget

**Returns:**
- `dict`: Allocation recommendations

##### `optimize_marketing_mix(channels, channel_costs, channel_reach, channel_conversion, total_budget=50000)`
Optimize marketing budget allocation across channels.

**Parameters:**
- `channels` (list): Channel names
- `channel_costs` (list): Cost per impression
- `channel_reach` (list): Maximum reach per channel
- `channel_conversion` (list): Conversion rates
- `total_budget` (float): Marketing budget

**Returns:**
- `dict`: Optimal budget allocation

## Data Types

All DataFrames returned by data generators include appropriate columns:

### Customer Churn Data
- `customer_id`, `age`, `tenure_months`, `monthly_charges`, `total_charges`
- `num_products`, `support_calls`, `contract_type`, `payment_method`
- `customer_satisfaction`, `churned` (target)

### Sales Forecast Data
- `date`, `sales`, `day_of_week`, `month`, `is_weekend`

### Fraud Detection Data
- `transaction_id`, `amount`, `hour_of_day`, `day_of_week`
- `merchant_category`, `distance_from_home`, `num_transactions_day`
- `avg_transaction_amount`, `account_age_days`, `is_fraud` (target)

### Product Recommendation Data
Interactions: `user_id`, `product_id`, `rating`, `purchased`
Products: `product_id`, `category`, `price`, `rating`, `popularity_score`

### Pricing Optimization Data
- `product_id`, `base_cost`, `base_price`, `price_elasticity`
- `max_demand`, `inventory`, `storage_cost`, `competitor_price`
