"""
Product Recommendation System Example

Demonstrates prescriptive analytics for personalized recommendations.
"""

import sys
sys.path.insert(0, '..')

from paddle.data.generator import DataGenerator
from paddle.models.prescriptive import PrescriptiveModels


def main():
    """Run product recommendation system example."""
    print("=" * 80)
    print("PADDLE - Product Recommendation System Example")
    print("=" * 80)
    print()
    
    # Generate synthetic recommendation data
    print("Step 1: Generating synthetic user-product interaction data...")
    generator = DataGenerator(random_state=42)
    interactions_df, products_df = generator.generate_product_recommendation_data(
        n_users=1000,
        n_products=500
    )
    print(f"Generated {len(interactions_df)} user-product interactions")
    print(f"Total users: {interactions_df['user_id'].nunique()}")
    print(f"Total products: {len(products_df)}")
    print()
    
    # Generate recommendations
    print("Step 2: Generating personalized recommendations for sample users...")
    model = PrescriptiveModels()
    
    sample_users = [1, 50, 100]
    
    for user_id in sample_users:
        print(f"\n{'=' * 80}")
        print(f"Recommendations for User {user_id}")
        print('=' * 80)
        
        # Get user history
        user_history = interactions_df[interactions_df['user_id'] == user_id]
        print(f"User has interacted with {len(user_history)} products")
        
        if len(user_history) > 0:
            # Get purchased products
            purchased = user_history[user_history['purchased']]
            if len(purchased) > 0:
                purchased_products = products_df[
                    products_df['product_id'].isin(purchased['product_id'])
                ]
                print(f"Purchased categories: {purchased_products['category'].value_counts().to_dict()}")
        
        # Generate recommendations
        recommendations = model.optimize_product_recommendations(
            interactions_df,
            products_df,
            user_id,
            n_recommendations=5
        )
        
        print(f"\nTop 5 Recommendations:")
        print(f"{'Product':>8} | {'Category':>15} | {'Price':>8} | {'Rating':>7} | {'Score':>8}")
        print("-" * 70)
        for rec in recommendations:
            print(f"{rec['product_id']:8d} | "
                  f"{rec['category']:>15s} | "
                  f"${rec['price']:7.2f} | "
                  f"{rec['rating']:7.2f} | "
                  f"{rec['recommendation_score']:8.2f}")
    
    print()
    print("=" * 80)
    print("Business Insights")
    print("=" * 80)
    print("- Personalized recommendations increase conversion rates")
    print("- Category preferences and ratings drive recommendation quality")
    print("- Actionable: Implement real-time recommendation engine on website")
    print("=" * 80)


if __name__ == "__main__":
    main()
