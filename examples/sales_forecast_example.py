"""
Sales Forecasting Example

Demonstrates time series forecasting for sales prediction.
"""

import sys
sys.path.insert(0, '..')

from paddle.data.generator import DataGenerator
from paddle.models.predictive import PredictiveModels


def main():
    """Run sales forecasting example."""
    print("=" * 80)
    print("PADDLE - Sales Forecasting Example")
    print("=" * 80)
    print()
    
    # Generate synthetic sales data
    print("Step 1: Generating synthetic sales time series data...")
    generator = DataGenerator(random_state=42)
    df = generator.generate_sales_forecast_data(n_periods=730)  # 2 years
    print(f"Generated {len(df)} days of sales data")
    print(f"Average daily sales: ${df['sales'].mean():.2f}")
    print()
    
    # Train sales forecasting model
    print("Step 2: Training sales forecasting model...")
    model = PredictiveModels(random_state=42)
    metrics = model.train_sales_forecast_model(df)
    print()
    
    # Display results
    print("=" * 80)
    print("Model Performance Metrics")
    print("=" * 80)
    print(f"RMSE (Root Mean Squared Error): ${metrics['rmse']:.2f}")
    print(f"MAE (Mean Absolute Error):      ${metrics['mae']:.2f}")
    print(f"R² Score:                        {metrics['r2_score']:.4f}")
    print()
    
    print("Top 5 Most Important Features:")
    feature_importance = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feature, importance in feature_importance:
        print(f"  {feature:30s}: {importance:.4f}")
    print()
    
    print("Sample Predictions vs Actuals (last 10 days):")
    print("  Actual    | Predicted | Difference")
    print("  " + "-" * 38)
    for actual, predicted in metrics['actual_vs_predicted'][-10:]:
        diff = predicted - actual
        print(f"  ${actual:7.2f} | ${predicted:9.2f} | ${diff:+9.2f}")
    print()
    
    print("=" * 80)
    print("Business Insights")
    print("=" * 80)
    print("- The model captures weekly and seasonal patterns in sales")
    print("- Historical sales patterns are strong predictors of future sales")
    print("- Actionable: Use forecasts for inventory planning and staffing")
    print("=" * 80)


if __name__ == "__main__":
    main()
