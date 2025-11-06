"""Tests for data generation module."""

import pytest
import pandas as pd
from paddle.data.generator import DataGenerator


class TestDataGenerator:
    """Test DataGenerator class."""
    
    def test_init(self):
        """Test DataGenerator initialization."""
        generator = DataGenerator(random_state=42)
        assert generator.random_state == 42
    
    def test_generate_customer_churn_data(self):
        """Test customer churn data generation."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_customer_churn_data(n_samples=1000)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert 'customer_id' in df.columns
        assert 'churned' in df.columns
        assert df['churned'].isin([0, 1]).all()
        assert 0 < df['churned'].mean() < 1
    
    def test_generate_sales_forecast_data(self):
        """Test sales forecast data generation."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_sales_forecast_data(n_periods=365)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 365
        assert 'date' in df.columns
        assert 'sales' in df.columns
        assert (df['sales'] >= 0).all()
    
    def test_generate_product_recommendation_data(self):
        """Test product recommendation data generation."""
        generator = DataGenerator(random_state=42)
        interactions_df, products_df = generator.generate_product_recommendation_data(
            n_users=100,
            n_products=50
        )
        
        assert isinstance(interactions_df, pd.DataFrame)
        assert isinstance(products_df, pd.DataFrame)
        assert len(products_df) == 50
        assert 'user_id' in interactions_df.columns
        assert 'product_id' in interactions_df.columns
        assert 'rating' in interactions_df.columns
    
    def test_generate_pricing_optimization_data(self):
        """Test pricing optimization data generation."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_pricing_optimization_data(n_products=50)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert 'product_id' in df.columns
        assert 'base_cost' in df.columns
        assert 'base_price' in df.columns
        assert (df['base_price'] > df['base_cost']).all()
    
    def test_generate_fraud_detection_data(self):
        """Test fraud detection data generation."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_fraud_detection_data(n_samples=5000)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5000
        assert 'transaction_id' in df.columns
        assert 'is_fraud' in df.columns
        assert df['is_fraud'].isin([0, 1]).all()
        assert 0 < df['is_fraud'].mean() < 0.5  # Fraud should be minority class
    
    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        generator1 = DataGenerator(random_state=42)
        generator2 = DataGenerator(random_state=42)
        
        df1 = generator1.generate_customer_churn_data(n_samples=100)
        df2 = generator2.generate_customer_churn_data(n_samples=100)
        
        pd.testing.assert_frame_equal(df1, df2)
