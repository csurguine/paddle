"""
Customer Churn Prediction Example

Demonstrates predictive modeling for customer churn using machine learning.
"""

import sys
sys.path.insert(0, '..')

from paddle.data.generator import DataGenerator
from paddle.models.predictive import PredictiveModels
import json


def main():
    """Run customer churn prediction example."""
    print("=" * 80)
    print("PADDLE - Customer Churn Prediction Example")
    print("=" * 80)
    print()
    
    # Generate synthetic customer data
    print("Step 1: Generating synthetic customer data...")
    generator = DataGenerator(random_state=42)
    df = generator.generate_customer_churn_data(n_samples=10000)
    print(f"Generated {len(df)} customer records")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print()
    
    # Train churn prediction model
    print("Step 2: Training churn prediction model...")
    model = PredictiveModels(random_state=42)
    metrics = model.train_churn_prediction_model(df)
    print()
    
    # Display results
    print("=" * 80)
    print("Model Performance Metrics")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
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
    
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Negatives:  {cm[0][0]:5d}  |  False Positives: {cm[0][1]:5d}")
    print(f"  False Negatives: {cm[1][0]:5d}  |  True Positives:  {cm[1][1]:5d}")
    print()
    
    print("=" * 80)
    print("Business Insights")
    print("=" * 80)
    print("- The model can identify customers at risk of churning with high accuracy")
    print("- Key churn indicators: support calls, tenure, and satisfaction")
    print("- Actionable: Target high-risk customers with retention campaigns")
    print("=" * 80)


if __name__ == "__main__":
    main()
