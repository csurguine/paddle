"""
Inventory Allocation Optimization Example

Demonstrates prescriptive analytics for optimal inventory decisions.
"""

import sys
sys.path.insert(0, '..')

from paddle.data.generator import DataGenerator
from paddle.models.prescriptive import PrescriptiveModels


def main():
    """Run inventory optimization example."""
    print("=" * 80)
    print("PADDLE - Inventory Allocation Optimization Example")
    print("=" * 80)
    print()
    
    # Generate synthetic pricing data
    print("Step 1: Generating synthetic product data...")
    generator = DataGenerator(random_state=42)
    df = generator.generate_pricing_optimization_data(n_products=100)
    print(f"Generated data for {len(df)} products")
    print()
    
    # Optimize inventory allocation
    total_budget = 100000
    print(f"Step 2: Optimizing inventory allocation with ${total_budget:,} budget...")
    model = PrescriptiveModels()
    results = model.optimize_inventory_allocation(df, total_budget=total_budget)
    print()
    
    # Display results
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"Total Budget:           ${results['total_budget']:,.2f}")
    print(f"Budget Used:            ${results['budget_used']:,.2f} ({results['budget_used']/results['total_budget']*100:.1f}%)")
    print(f"Budget Remaining:       ${results['budget_remaining']:,.2f}")
    print()
    print(f"Total Units to Stock:   {results['total_units']:,.0f}")
    print(f"Expected Revenue:       ${results['total_expected_revenue']:,.2f}")
    print(f"Expected Profit:        ${results['total_expected_profit']:,.2f}")
    print(f"Average ROI:            {results['average_roi']:.2%}")
    print()
    
    print("Top 10 Products by ROI:")
    sorted_products = sorted(
        results['detailed_allocation'],
        key=lambda x: x['roi'],
        reverse=True
    )[:10]
    
    print(f"{'Product':>8} | {'Units':>7} | {'Cost $':>10} | {'Revenue $':>12} | {'Profit $':>11} | {'ROI':>8}")
    print("-" * 80)
    for product in sorted_products:
        print(f"{int(product['product_id']):8d} | "
              f"{int(product['units_to_stock']):7d} | "
              f"${product['total_cost']:9.2f} | "
              f"${product['expected_revenue']:11.2f} | "
              f"${product['expected_profit']:10.2f} | "
              f"{product['roi']:7.2%}")
    print()
    
    print("=" * 80)
    print("Business Insights")
    print("=" * 80)
    roi_value = results['average_roi'] * 100
    print(f"- Optimized allocation achieves {roi_value:.1f}% average ROI")
    print("- High-ROI products are prioritized within budget constraints")
    print("- Expected profit: ${:,.2f}".format(results['total_expected_profit']))
    print("- Actionable: Focus inventory investment on high-performing products")
    print("=" * 80)


if __name__ == "__main__":
    main()
