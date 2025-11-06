"""
Fraud Detection Example

Demonstrates anomaly detection for financial transaction fraud.
"""

import sys
sys.path.insert(0, '..')

from paddle.data.generator import DataGenerator
from paddle.models.predictive import PredictiveModels


def main():
    """Run fraud detection example."""
    print("=" * 80)
    print("PADDLE - Fraud Detection Example")
    print("=" * 80)
    print()
    
    # Generate synthetic transaction data
    print("Step 1: Generating synthetic transaction data...")
    generator = DataGenerator(random_state=42)
    df = generator.generate_fraud_detection_data(n_samples=50000)
    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print()
    
    # Train fraud detection model
    print("Step 2: Training fraud detection model...")
    model = PredictiveModels(random_state=42)
    metrics = model.train_fraud_detection_model(df)
    print()
    
    # Display results
    print("=" * 80)
    print("Model Performance Metrics")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (of flagged transactions, % that are actual fraud)")
    print(f"Recall:    {metrics['recall']:.4f} (of actual fraud, % that are detected)")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print()
    
    print("Top 5 Fraud Indicators:")
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
    print("- High-precision fraud detection minimizes false positives")
    print("- Key fraud indicators: transaction amount, time, and frequency")
    print("- Actionable: Real-time fraud scoring for transaction monitoring")
    print(f"- Estimated savings: Detect {metrics['recall']:.1%} of fraud cases early")
    print("=" * 80)


if __name__ == "__main__":
    main()
