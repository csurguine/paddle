"""
Prescriptive Models Module

Implement prescriptive analytics for optimization and decision-making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class PrescriptiveModels:
    """Implement prescriptive modeling techniques for optimization."""
    
    def __init__(self):
        """Initialize prescriptive models."""
        self.solutions = {}
    
    def optimize_pricing(
        self,
        df: pd.DataFrame,
        objective: str = 'maximize_profit'
    ) -> Dict[str, Any]:
        """
        Optimize product pricing to maximize profit or revenue.
        
        Args:
            df: DataFrame with product pricing data
            objective: Objective function ('maximize_profit' or 'maximize_revenue')
            
        Returns:
            Dictionary with optimal prices and expected outcomes
        """
        results = []
        
        for _, row in df.iterrows():
            product_id = row['product_id']
            base_cost = row['base_cost']
            base_price = row['base_price']
            elasticity = row['price_elasticity']
            max_demand = row['max_demand']
            inventory = row['inventory']
            
            # Search for optimal price
            best_price = base_price
            best_objective = -np.inf
            
            # Test prices from 120% to 180% of base cost
            for price_multiplier in np.linspace(1.2, 1.8, 100):
                test_price = base_cost * price_multiplier
                
                # Calculate demand at test price using price elasticity
                price_change_pct = (test_price - base_price) / base_price
                demand = max_demand * (1 + elasticity * price_change_pct)
                demand = max(0, min(demand, inventory))  # Constrain by inventory
                
                # Calculate objective
                if objective == 'maximize_profit':
                    objective_value = (test_price - base_cost) * demand
                else:  # maximize_revenue
                    objective_value = test_price * demand
                
                if objective_value > best_objective:
                    best_objective = objective_value
                    best_price = test_price
            
            # Calculate metrics at optimal price
            optimal_demand = max_demand * (1 + elasticity * (best_price - base_price) / base_price)
            optimal_demand = max(0, min(optimal_demand, inventory))
            
            results.append({
                'product_id': product_id,
                'current_price': base_price,
                'optimal_price': best_price,
                'price_change_pct': ((best_price - base_price) / base_price) * 100,
                'expected_demand': optimal_demand,
                'expected_revenue': best_price * optimal_demand,
                'expected_profit': (best_price - base_cost) * optimal_demand
            })
        
        results_df = pd.DataFrame(results)
        
        summary = {
            'total_current_revenue': (df['base_price'] * df['expected_demand']).sum(),
            'total_optimal_revenue': results_df['expected_revenue'].sum(),
            'total_current_profit': ((df['base_price'] - df['base_cost']) * df['expected_demand']).sum(),
            'total_optimal_profit': results_df['expected_profit'].sum(),
            'avg_price_increase_pct': results_df['price_change_pct'].mean(),
            'detailed_results': results_df.to_dict('records')
        }
        
        self.solutions['pricing'] = summary
        return summary
    
    def optimize_product_recommendations(
        self,
        interactions_df: pd.DataFrame,
        products_df: pd.DataFrame,
        user_id: int,
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized product recommendations.
        
        Args:
            interactions_df: User-product interaction data
            products_df: Product features data
            user_id: User ID to generate recommendations for
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommended products with scores
        """
        # Get user's interaction history
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            # Cold start: recommend popular products
            recommendations = products_df.nlargest(n_recommendations, 'popularity_score')
        else:
            # Get user's preferred categories
            user_products = user_interactions['product_id'].values
            user_product_details = products_df[products_df['product_id'].isin(user_products)]
            preferred_categories = user_product_details['category'].value_counts().index.tolist()
            
            # Get user's average rating
            avg_user_rating = user_interactions['rating'].mean()
            
            # Calculate scores for all products
            scores = []
            for _, product in products_df.iterrows():
                if product['product_id'] in user_products:
                    continue  # Skip already purchased products
                
                score = 0
                
                # Category preference
                if product['category'] in preferred_categories:
                    category_rank = preferred_categories.index(product['category'])
                    score += (len(preferred_categories) - category_rank) * 20
                
                # Product rating
                score += product['rating'] * 10
                
                # Popularity
                score += product['popularity_score'] * 15
                
                # Price consideration (prefer mid-range)
                avg_price = products_df['price'].mean()
                price_deviation = abs(product['price'] - avg_price) / avg_price
                score -= price_deviation * 10
                
                scores.append({
                    'product_id': product['product_id'],
                    'category': product['category'],
                    'price': product['price'],
                    'rating': product['rating'],
                    'recommendation_score': score
                })
            
            # Sort by score and get top N
            scores_df = pd.DataFrame(scores)
            recommendations = scores_df.nlargest(n_recommendations, 'recommendation_score')
        
        return recommendations.to_dict('records')
    
    def optimize_inventory_allocation(
        self,
        df: pd.DataFrame,
        total_budget: float = 100000
    ) -> Dict[str, Any]:
        """
        Optimize inventory allocation based on budget constraints.
        
        Args:
            df: DataFrame with product data
            total_budget: Total budget available for inventory
            
        Returns:
            Dictionary with optimal allocation
        """
        # Calculate expected profit per unit
        df['profit_per_unit'] = df['base_price'] - df['base_cost']
        df['expected_profit'] = df['profit_per_unit'] * df['expected_demand']
        df['roi'] = df['expected_profit'] / (df['base_cost'] * df['expected_demand'])
        
        # Sort by ROI
        df_sorted = df.sort_values('roi', ascending=False)
        
        allocation = []
        remaining_budget = total_budget
        
        for _, product in df_sorted.iterrows():
            # Calculate how many units we can afford
            cost_per_unit = product['base_cost']
            max_units = min(
                int(remaining_budget / cost_per_unit),
                int(product['expected_demand']),
                product['inventory']
            )
            
            if max_units > 0:
                total_cost = max_units * cost_per_unit
                expected_revenue = max_units * product['base_price']
                expected_profit = max_units * product['profit_per_unit']
                
                allocation.append({
                    'product_id': product['product_id'],
                    'units_to_stock': max_units,
                    'total_cost': total_cost,
                    'expected_revenue': expected_revenue,
                    'expected_profit': expected_profit,
                    'roi': product['roi']
                })
                
                remaining_budget -= total_cost
        
        allocation_df = pd.DataFrame(allocation)
        
        summary = {
            'total_budget': total_budget,
            'budget_used': total_budget - remaining_budget,
            'budget_remaining': remaining_budget,
            'total_units': allocation_df['units_to_stock'].sum(),
            'total_expected_revenue': allocation_df['expected_revenue'].sum(),
            'total_expected_profit': allocation_df['expected_profit'].sum(),
            'average_roi': allocation_df['roi'].mean(),
            'detailed_allocation': allocation_df.to_dict('records')
        }
        
        self.solutions['inventory'] = summary
        return summary
    
    def optimize_marketing_mix(
        self,
        channels: List[str],
        channel_costs: List[float],
        channel_reach: List[int],
        channel_conversion: List[float],
        total_budget: float = 50000
    ) -> Dict[str, Any]:
        """
        Optimize marketing budget allocation across channels.
        
        Args:
            channels: List of marketing channel names
            channel_costs: Cost per impression for each channel
            channel_reach: Maximum reach for each channel
            channel_conversion: Conversion rate for each channel
            total_budget: Total marketing budget
            
        Returns:
            Dictionary with optimal allocation
        """
        # Calculate expected value per dollar for each channel
        allocation = []
        remaining_budget = total_budget
        
        channel_data = list(zip(channels, channel_costs, channel_reach, channel_conversion))
        # Sort by expected value (conversions per dollar)
        channel_data_sorted = sorted(
            channel_data,
            key=lambda x: x[3] / x[1] if x[1] > 0 else 0,
            reverse=True
        )
        
        for channel, cost, max_reach, conversion in channel_data_sorted:
            # Calculate optimal spend for this channel
            max_spend = min(remaining_budget, cost * max_reach)
            impressions = int(max_spend / cost) if cost > 0 else 0
            expected_conversions = impressions * conversion
            
            if impressions > 0:
                allocation.append({
                    'channel': channel,
                    'budget_allocated': impressions * cost,
                    'impressions': impressions,
                    'expected_conversions': expected_conversions,
                    'cost_per_conversion': (impressions * cost) / expected_conversions if expected_conversions > 0 else float('inf')
                })
                
                remaining_budget -= impressions * cost
        
        allocation_df = pd.DataFrame(allocation)
        
        summary = {
            'total_budget': total_budget,
            'budget_used': total_budget - remaining_budget,
            'total_impressions': allocation_df['impressions'].sum(),
            'total_expected_conversions': allocation_df['expected_conversions'].sum(),
            'average_cost_per_conversion': (total_budget - remaining_budget) / allocation_df['expected_conversions'].sum() if allocation_df['expected_conversions'].sum() > 0 else float('inf'),
            'detailed_allocation': allocation_df.to_dict('records')
        }
        
        self.solutions['marketing'] = summary
        return summary
