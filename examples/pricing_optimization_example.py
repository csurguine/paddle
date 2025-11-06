"""
Pricing Optimization Example

Demonstrates prescriptive analytics for optimal pricing decisions.
"""

import sys
sys.path.insert(0, '..')

from paddle.data.generator import DataGenerator
from paddle.models.prescriptive import PrescriptiveModels


def main():
    """Run pricing optimization example."""
    print("=" * 80)
    print("PADDLE - Pricing Optimization Example")
    print("=" * 80)
    print()
    
    # Generate synthetic pricing data
    print("Step 1: Generating synthetic product pricing data...")
    generator = DataGenerator(random_state=42)
    df = generator.generate_pricing_optimization_data(n_products=50)
    print(f"Generated pricing data for {len(df)} products")
    print()
    
    # Optimize pricing
    print("Step 2: Optimizing prices to maximize profit...")
    model = PrescriptiveModels()
    results = model.optimize_pricing(df, objective='maximize_profit')
    print()
    
    # Display results
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"Current Total Revenue: ${results['total_current_revenue']:,.2f}")
    print(f"Optimal Total Revenue: ${results['total_optimal_revenue']:,.2f}")
    print(f"Revenue Increase:      ${results['total_optimal_revenue'] - results['total_current_revenue']:,.2f}")
    print()
    print(f"Current Total Profit:  ${results['total_current_profit']:,.2f}")
    print(f"Optimal Total Profit:  ${results['total_optimal_profit']:,.2f}")
    print(f"Profit Increase:       ${results['total_optimal_profit'] - results['total_current_profit']:,.2f}")
    print(f"Average Price Change:  {results['avg_price_increase_pct']:.2f}%")
    print()
    
    print("Sample Product Recommendations (Top 5 by profit improvement):")
    sorted_products = sorted(
        results['detailed_results'],
        key=lambda x: x['expected_profit'],
        reverse=True
    )[:5]
    
    print(f"{'Product':>8} | {'Current $':>10} | {'Optimal $':>10} | {'Change':>8} | {'Profit $':>10}")
    print("-" * 70)
    for product in sorted_products:
        print(f"{int(product['product_id']):8d} | "
              f"${product['current_price']:9.2f} | "
              f"${product['optimal_price']:9.2f} | "
              f"{product['price_change_pct']:7.2f}% | "
              f"${product['expected_profit']:9.2f}")
    print()
    
    print("=" * 80)
    print("Business Insights")
    print("=" * 80)
    profit_increase_pct = ((results['total_optimal_profit'] - results['total_current_profit']) / 
                           results['total_current_profit'] * 100)
    print(f"- Optimized pricing can increase profit by {profit_increase_pct:.1f}%")
    print("- Price elasticity drives optimal pricing strategy")
    print("- Actionable: Implement dynamic pricing based on demand patterns")
    print("=" * 80)


if __name__ == "__main__":
    main()
