"""
Data Generation Module

Generate synthetic large-scale datasets for various business problems.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class DataGenerator:
    """Generate synthetic datasets for different business scenarios."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_customer_churn_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate customer churn dataset for predictive modeling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with customer features and churn label
        """
        np.random.seed(self.random_state)
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'tenure_months': np.random.randint(1, 120, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 10000, n_samples),
            'num_products': np.random.randint(1, 6, n_samples),
            'support_calls': np.random.poisson(2, n_samples),
            'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], n_samples),
            'customer_satisfaction': np.random.uniform(1, 5, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate churn based on features (logical relationship)
        churn_probability = (
            0.1 +
            0.2 * (df['support_calls'] > 3).astype(int) +
            0.15 * (df['tenure_months'] < 12).astype(int) +
            0.1 * (df['customer_satisfaction'] < 2.5).astype(int) +
            0.15 * (df['contract_type'] == 'Month-to-Month').astype(int) -
            0.1 * (df['num_products'] > 2).astype(int)
        )
        churn_probability = np.clip(churn_probability, 0, 1)
        df['churned'] = (np.random.random(n_samples) < churn_probability).astype(int)
        
        return df
    
    def generate_sales_forecast_data(self, n_periods: int = 730) -> pd.DataFrame:
        """
        Generate time series data for sales forecasting.
        
        Args:
            n_periods: Number of time periods (days)
            
        Returns:
            DataFrame with date and sales data
        """
        np.random.seed(self.random_state)
        
        dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='D')
        
        # Generate trend component
        trend = np.linspace(1000, 2000, n_periods)
        
        # Generate seasonal component (weekly and yearly)
        weekly_season = 200 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
        yearly_season = 300 * np.sin(2 * np.pi * np.arange(n_periods) / 365)
        
        # Add noise
        noise = np.random.normal(0, 50, n_periods)
        
        # Combine components
        sales = trend + weekly_season + yearly_season + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative sales
        
        df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': dates.dayofweek.isin([5, 6]).astype(int)
        })
        
        return df
    
    def generate_product_recommendation_data(
        self, 
        n_users: int = 1000, 
        n_products: int = 500
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate user-product interaction data for recommendation systems.
        
        Args:
            n_users: Number of users
            n_products: Number of products
            
        Returns:
            Tuple of (interactions_df, products_df)
        """
        np.random.seed(self.random_state)
        
        # Generate product features
        products_df = pd.DataFrame({
            'product_id': range(1, n_products + 1),
            'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_products),
            'price': np.random.uniform(10, 500, n_products),
            'rating': np.random.uniform(2.5, 5, n_products),
            'popularity_score': np.random.uniform(0, 1, n_products)
        })
        
        # Generate user-product interactions
        n_interactions = n_users * 10  # Average 10 interactions per user
        interactions = []
        
        for _ in range(n_interactions):
            user_id = np.random.randint(1, n_users + 1)
            product_id = np.random.randint(1, n_products + 1)
            
            # Rating influenced by product rating
            base_rating = products_df.loc[product_id - 1, 'rating']
            rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)
            
            interactions.append({
                'user_id': user_id,
                'product_id': product_id,
                'rating': rating,
                'purchased': np.random.random() < 0.3
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        return interactions_df, products_df
    
    def generate_pricing_optimization_data(self, n_products: int = 100) -> pd.DataFrame:
        """
        Generate data for pricing optimization problems.
        
        Args:
            n_products: Number of products
            
        Returns:
            DataFrame with product pricing and demand data
        """
        np.random.seed(self.random_state)
        
        data = {
            'product_id': range(1, n_products + 1),
            'base_cost': np.random.uniform(10, 100, n_products),
            'base_price': np.random.uniform(20, 200, n_products),
            'price_elasticity': np.random.uniform(-2.5, -0.5, n_products),
            'max_demand': np.random.randint(100, 1000, n_products),
            'inventory': np.random.randint(50, 500, n_products),
            'storage_cost': np.random.uniform(0.5, 5, n_products),
            'competitor_price': np.random.uniform(15, 180, n_products)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate expected demand based on price elasticity
        df['expected_demand'] = df['max_demand'] * (
            1 + df['price_elasticity'] * (df['base_price'] - df['base_cost']) / df['base_cost']
        )
        df['expected_demand'] = np.maximum(df['expected_demand'], 0)
        
        return df
    
    def generate_fraud_detection_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Generate transaction data for fraud detection.
        
        Args:
            n_samples: Number of transactions
            
        Returns:
            DataFrame with transaction features and fraud label
        """
        np.random.seed(self.random_state)
        
        data = {
            'transaction_id': range(1, n_samples + 1),
            'amount': np.random.lognormal(4, 2, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'merchant_category': np.random.choice(['Retail', 'Food', 'Travel', 'Entertainment', 'Online'], n_samples),
            'distance_from_home': np.random.exponential(20, n_samples),
            'num_transactions_day': np.random.poisson(3, n_samples),
            'avg_transaction_amount': np.random.uniform(20, 200, n_samples),
            'account_age_days': np.random.randint(1, 3650, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate fraud based on suspicious patterns
        fraud_probability = (
            0.01 +
            0.15 * (df['amount'] > 1000).astype(int) +
            0.1 * (df['hour_of_day'].isin([1, 2, 3, 4])).astype(int) +
            0.08 * (df['distance_from_home'] > 100).astype(int) +
            0.12 * (df['num_transactions_day'] > 8).astype(int) +
            0.05 * (df['account_age_days'] < 30).astype(int)
        )
        fraud_probability = np.clip(fraud_probability, 0, 1)
        df['is_fraud'] = (np.random.random(n_samples) < fraud_probability).astype(int)
        
        return df
