"""Tests for prescriptive models module."""

import pytest
import pandas as pd
from paddle.data.generator import DataGenerator
from paddle.models.prescriptive import PrescriptiveModels


class TestPrescriptiveModels:
    """Test PrescriptiveModels class."""
    
    def test_init(self):
        """Test PrescriptiveModels initialization."""
        model = PrescriptiveModels()
        assert isinstance(model.solutions, dict)
    
    def test_optimize_pricing(self):
        """Test pricing optimization."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_pricing_optimization_data(n_products=20)
        
        model = PrescriptiveModels()
        results = model.optimize_pricing(df, objective='maximize_profit')
        
        assert 'total_optimal_profit' in results
        assert 'total_current_profit' in results
        assert 'detailed_results' in results
        assert len(results['detailed_results']) == 20
        assert results['total_optimal_profit'] >= results['total_current_profit']
    
    def test_optimize_product_recommendations(self):
        """Test product recommendation generation."""
        generator = DataGenerator(random_state=42)
        interactions_df, products_df = generator.generate_product_recommendation_data(
            n_users=100,
            n_products=50
        )
        
        model = PrescriptiveModels()
        recommendations = model.optimize_product_recommendations(
            interactions_df,
            products_df,
            user_id=1,
            n_recommendations=5
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        if len(recommendations) > 0:
            assert 'product_id' in recommendations[0]
            assert 'recommendation_score' in recommendations[0]
    
    def test_optimize_inventory_allocation(self):
        """Test inventory allocation optimization."""
        generator = DataGenerator(random_state=42)
        df = generator.generate_pricing_optimization_data(n_products=30)
        
        model = PrescriptiveModels()
        results = model.optimize_inventory_allocation(df, total_budget=50000)
        
        assert 'total_budget' in results
        assert 'budget_used' in results
        assert 'total_expected_profit' in results
        assert results['budget_used'] <= results['total_budget']
        assert results['total_expected_profit'] > 0
    
    def test_optimize_marketing_mix(self):
        """Test marketing mix optimization."""
        model = PrescriptiveModels()
        results = model.optimize_marketing_mix(
            channels=['Social Media', 'Email', 'Search', 'Display'],
            channel_costs=[0.5, 0.1, 1.0, 0.3],
            channel_reach=[100000, 50000, 80000, 120000],
            channel_conversion=[0.02, 0.05, 0.03, 0.01],
            total_budget=10000
        )
        
        assert 'total_budget' in results
        assert 'budget_used' in results
        assert 'total_expected_conversions' in results
        assert results['budget_used'] <= results['total_budget']
        assert results['total_expected_conversions'] > 0
