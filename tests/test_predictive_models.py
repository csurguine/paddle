"""Tests for predictive models module."""

import pytest
import pandas as pd
from paddle.data.generator import DataGenerator
from paddle.models.predictive import PredictiveModels


class TestPredictiveModels:
    """Test PredictiveModels class."""
    
    def test_init(self):
        """Test PredictiveModels initialization."""
        model = PredictiveModels(random_state=42)
        assert model.random_state == 42
        assert isinstance(model.models, dict)
        assert isinstance(model.scalers, dict)
    
    def test_train_churn_prediction_model(self):
        """Test churn prediction model training."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_customer_churn_data(n_samples=1000)
        
        model = PredictiveModels(random_state=42)
        metrics = model.train_churn_prediction_model(df)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 'churn' in model.models
    
    def test_train_sales_forecast_model(self):
        """Test sales forecasting model training."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_sales_forecast_data(n_periods=200)
        
        model = PredictiveModels(random_state=42)
        metrics = model.train_sales_forecast_model(df)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert metrics['rmse'] > 0
        assert -1 <= metrics['r2_score'] <= 1
        assert 'sales_forecast' in model.models
    
    def test_train_fraud_detection_model(self):
        """Test fraud detection model training."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_fraud_detection_data(n_samples=5000)
        
        model = PredictiveModels(random_state=42)
        metrics = model.train_fraud_detection_model(df)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 'fraud' in model.models
    
    def test_model_persistence(self):
        """Test that trained models are stored."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_customer_churn_data(n_samples=500)
        
        model = PredictiveModels(random_state=42)
        model.train_churn_prediction_model(df)
        
        assert 'churn' in model.models
        assert 'churn' in model.scalers
        assert model.models['churn'] is not None
